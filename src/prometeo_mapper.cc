#include <array>
#include <deque>
#include <iostream>
#include <fstream>
#include <regex>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "mappers/default_mapper.h"
#include "realm/logging.h"

#include "config_schema.h"
#include "prometeo_mapper.h"

using namespace Legion;
using namespace Legion::Mapping;

//=============================================================================
// DOCUMENTATION
//=============================================================================

// Assume we're running 2 CPU-only samples, A and B, tiled as follows:
//   tiles(A) = [1,2,1], tilesPerRank(A) = [1,1,1]
//   tiles(B) = [6,1,1], tilesPerRank(B) = [3,1,1]
// Based on this configuration, we calculate the following:
//   #shards(A) = 2, #splintersPerShard(A) = 1
//   #shards(B) = 2, #splintersPerShard(B) = 3
// Each shard is placed on a separate rank in row-major order, so we will need
// 4 ranks in total. The splinters within each shard are allocated in
// round-robin, row-major order to the processors on the corresponding rank
// (some processors may receive more than 1 splinter). Assume each rank has 2
// CPU processors. Then the mapping will be as follows:
//   Sample    Tile -> Shard Splinter -> Rank CPU
//   --------------------------------------------
//        A [0,0,0] ->     0        0 ->   0    0
//   --------------------------------------------
//        A [0,1,0] ->     1        0 ->   1    0
//   --------------------------------------------
//        B [0,0,0] ->     0        0 ->   2    0
//        B [1,0,0] ->     0        1 ->   2    1
//        B [2,0,0] ->     0        2 ->   2    0
//   --------------------------------------------
//        B [3,0,0] ->     1        0 ->   3    0
//        B [4,0,0] ->     1        1 ->   3    1
//        B [5,0,0] ->     1        2 ->   3    0

//=============================================================================
// HELPER CODE
//=============================================================================

static Realm::Logger LOG("prometeo_mapper");

#define CHECK(cond, ...)                        \
   do {                                         \
      if (!(cond)) {                            \
         LOG.error(__VA_ARGS__);                \
         exit(1);                               \
      }                                         \
   } while(0)

#define EQUALS(s1, s2) (strcmp((s1), (s2)) == 0)

#define STARTS_WITH(str, prefix)                \
   (strncmp((str), (prefix), sizeof(prefix) - 1) == 0)

static const void* first_arg(const Task& task) {
   const char* ptr = static_cast<const char*>(task.args);
   // Skip over Regent-added arguments.
   // XXX: This assumes Regent's calling convention won't change.
   return static_cast<const void*>(ptr + sizeof(uint64_t));
}

//=============================================================================
// INTRA-SAMPLE MAPPING
//=============================================================================

typedef unsigned SplinterID;

class SampleMapping;

class SplinteringFunctor : public ShardingFunctor {
private:
   static ShardingID NEXT_ID;
public:
   SplinteringFunctor(Runtime* rt, SampleMapping& parent)
      : id(NEXT_ID++), parent_(parent) {
      rt->register_sharding_functor(id, this, true);
   }
public:
   AddressSpace get_rank(const DomainPoint &point);
   virtual SplinterID splinter(const DomainPoint &point) = 0;
public:
   const ShardingID id;
protected:
   SampleMapping& parent_;
};

ShardingID SplinteringFunctor::NEXT_ID = 12345;

class SampleMapping {
public:
   class Tiling3DFunctor;
   class Tiling2DFunctor;
   class HardcodedFunctor;
   class RankTiling3DFunctor;
   class RankTiling2DFunctor;
   class RankHardcodedFunctor;

public:
   SampleMapping(Runtime* rt, const Config& config, AddressSpace first_rank)
      : tiles_per_rank_{static_cast<unsigned>(config.Mapping.tilesPerRank[0]),
                        static_cast<unsigned>(config.Mapping.tilesPerRank[1]),
                        static_cast<unsigned>(config.Mapping.tilesPerRank[2])},
      ranks_per_dim_{   static_cast<unsigned>(config.Mapping.tiles[0]
                                            / config.Mapping.tilesPerRank[0]),
                        static_cast<unsigned>(config.Mapping.tiles[1]
                                            / config.Mapping.tilesPerRank[1]),
                        static_cast<unsigned>(config.Mapping.tiles[2]
                                            / config.Mapping.tilesPerRank[2])},
      first_rank_(first_rank),
      tiling_3d_functor_(new Tiling3DFunctor(rt, *this)),
      tiling_2d_functors_{{new Tiling2DFunctor(rt, *this, 0, false),
                           new Tiling2DFunctor(rt, *this, 0, true )},
                          {new Tiling2DFunctor(rt, *this, 1, false),
                           new Tiling2DFunctor(rt, *this, 1, true )},
                          {new Tiling2DFunctor(rt, *this, 2, false),
                           new Tiling2DFunctor(rt, *this, 2, true )}},
      rank_tiling_3d_functor_{new RankTiling3DFunctor(rt, *this, false),
                              new RankTiling3DFunctor(rt, *this, true)},
      rank_tiling_2d_functors_{{new RankTiling2DFunctor(rt, *this, 0, false),
                                new RankTiling2DFunctor(rt, *this, 0, true )},
                               {new RankTiling2DFunctor(rt, *this, 1, false),
                                new RankTiling2DFunctor(rt, *this, 1, true )},
                               {new RankTiling2DFunctor(rt, *this, 2, false),
                                new RankTiling2DFunctor(rt, *this, 2, true )}},
      lowpriority_(false) {
      for (unsigned x = 0; x < x_tiles(); ++x) {
         for (unsigned y = 0; y < y_tiles(); ++y) {
            for (unsigned z = 0; z < z_tiles(); ++z) {
               hardcoded_functors_.push_back(new HardcodedFunctor(rt, *this, Point<3>(x,y,z)));
            }
         }
      }
      for (unsigned x = 0; x < ranks_per_dim_[0]; ++x) {
         for (unsigned y = 0; y < ranks_per_dim_[1]; ++y) {
            for (unsigned z = 0; z < ranks_per_dim_[2]; ++z) {
               rank_hardcoded_functors_.push_back(new RankHardcodedFunctor(rt, *this, Point<3>(x,y,z)));
            }
         }
      }
   }

   SampleMapping(const SampleMapping& rhs) = delete;
   SampleMapping& operator=(const SampleMapping& rhs) = delete;

public:
   AddressSpace get_rank(ShardID shard_id) const {
      return first_rank_ + shard_id;
   }

   unsigned num_ranks() const {
      return ranks_per_dim_[0] * ranks_per_dim_[1] * ranks_per_dim_[2];
   }

   unsigned x_tiles() const {
      return tiles_per_rank_[0] * ranks_per_dim_[0];
   }

   unsigned y_tiles() const {
      return tiles_per_rank_[1] * ranks_per_dim_[1];
   }

   unsigned z_tiles() const {
      return tiles_per_rank_[2] * ranks_per_dim_[2];
   }

   unsigned num_tiles() const {
      return x_tiles() * y_tiles() * z_tiles();
   }

   Tiling3DFunctor* tiling_3d_functor() {
      return tiling_3d_functor_;
   }

   Tiling2DFunctor* tiling_2d_functor(int dim, bool dir) {
      assert(0 <= dim && dim < 3);
      return tiling_2d_functors_[dim][dir];
   }

   HardcodedFunctor* hardcoded_functor(const DomainPoint& tile) {
      assert(tile.get_dim() == 3);
      assert(0 <= tile[0] && tile[0] < x_tiles());
      assert(0 <= tile[1] && tile[1] < y_tiles());
      assert(0 <= tile[2] && tile[2] < z_tiles());
      return hardcoded_functors_[tile[0] * y_tiles() * z_tiles() +
                                 tile[1] * z_tiles() +
                                 tile[2]];
   }

   RankTiling3DFunctor* rank_tiling_3d_functor(bool fold = false) {
      return rank_tiling_3d_functor_[fold];
   }

   RankTiling2DFunctor* rank_tiling_2d_functor(int dim, bool dir) {
      assert(0 <= dim && dim < 3);
      return rank_tiling_2d_functors_[dim][dir];
   }

   RankHardcodedFunctor* rank_hardcoded_functor(const DomainPoint& tile) {
      assert(tile.get_dim() == 3);
      assert(0 <= tile[0] && tile[0] < ranks_per_dim_[0]);
      assert(0 <= tile[1] && tile[1] < ranks_per_dim_[1]);
      assert(0 <= tile[2] && tile[2] < ranks_per_dim_[2]);
      return rank_hardcoded_functors_[tile[0] * ranks_per_dim_[1] * ranks_per_dim_[2] +
                                      tile[1] * ranks_per_dim_[2] +
                                      tile[2]];
   }

   bool isLowpriority() const {
        return lowpriority_;
   }

   void setLowpriority() {
        lowpriority_ = true;
   }

public:
   // Maps tasks in a 3D index space launch according to the default tiling
   // logic (see description above).
   class Tiling3DFunctor : public SplinteringFunctor {
   public:
         Tiling3DFunctor(Runtime* rt, SampleMapping& parent)
            : SplinteringFunctor(rt, parent) {}
   public:
      virtual ShardID shard(const DomainPoint& point,
                            const Domain& full_space,
                            const size_t total_shards) {
         assert(point.get_dim() == 3);
         CHECK(0 <= point[0] && point[0] < parent_.x_tiles() &&
               0 <= point[1] && point[1] < parent_.y_tiles() &&
               0 <= point[2] && point[2] < parent_.z_tiles(),
               "Unexpected point on index space launch");
         return (point[0] / parent_.tiles_per_rank_[0]) * parent_.ranks_per_dim_[1]
                                                        * parent_.ranks_per_dim_[2] +
                (point[1] / parent_.tiles_per_rank_[1]) * parent_.ranks_per_dim_[2] +
                (point[2] / parent_.tiles_per_rank_[2]);
      }

      virtual SplinterID splinter(const DomainPoint &point) {
         assert(point.get_dim() == 3);
         CHECK(0 <= point[0] && point[0] < parent_.x_tiles() &&
               0 <= point[1] && point[1] < parent_.y_tiles() &&
               0 <= point[2] && point[2] < parent_.z_tiles(),
               "Unexpected point on index space launch");
         return (point[0] % parent_.tiles_per_rank_[0]) * parent_.tiles_per_rank_[1]
                                                        * parent_.tiles_per_rank_[2] +
                (point[1] % parent_.tiles_per_rank_[1]) * parent_.tiles_per_rank_[2] +
                (point[2] % parent_.tiles_per_rank_[2]);
      }
   };

   // Maps tasks in a 2D index space launch, by extending each domain point to a
   // 3D tile and deferring to the default strategy.
   // Parameter `dim` controls which dimension to add.
   // Parameter `dir` controls which extreme of that dimension to set.
   class Tiling2DFunctor : public SplinteringFunctor {
   public:
      Tiling2DFunctor(Runtime* rt, SampleMapping& parent,
                      unsigned dim, bool dir)
         : SplinteringFunctor(rt, parent), dim_(dim), dir_(dir) {}

   public:
      virtual ShardID shard(const DomainPoint& point,
                            const Domain& full_space,
                            const size_t total_shards) {
         return parent_.tiling_3d_functor_->shard
                (to_point_3d(point), full_space, total_shards);
      }

      virtual SplinterID splinter(const DomainPoint &point) {
         return parent_.tiling_3d_functor_->splinter(to_point_3d(point));
      }

   private:
      DomainPoint to_point_3d(const DomainPoint& point) const {
         assert(point.get_dim() == 2);
         unsigned coord =
            (dim_ == 0) ? (dir_ ? 0 : parent_.x_tiles()-1) :
            (dim_ == 1) ? (dir_ ? 0 : parent_.y_tiles()-1) :
           /*dim_ == 2*/  (dir_ ? 0 : parent_.z_tiles()-1) ;
         return
            (dim_ == 0) ? Point<3>(coord, point[0], point[1]) :
            (dim_ == 1) ? Point<3>(point[0], coord, point[1]) :
           /*dim_ == 2*/  Point<3>(point[0], point[1], coord) ;
      }

   private:
      unsigned dim_;
      bool dir_;
   };

   // Maps every task to the same shard & splinter (the ones corresponding to
   // the tile specified in the constructor).
   class HardcodedFunctor : public SplinteringFunctor {
   public:
      HardcodedFunctor(Runtime* rt,
                       SampleMapping& parent,
                       const DomainPoint& tile)
         : SplinteringFunctor(rt, parent), tile_(tile) {}
   public:
      virtual ShardID shard(const DomainPoint& point,
                            const Domain& full_space,
                            const size_t total_shards) {
         return parent_.tiling_3d_functor_->shard(tile_, full_space, total_shards);
      }

      virtual SplinterID splinter(const DomainPoint &point) {
         return parent_.tiling_3d_functor_->splinter(tile_);
      }
   private:
      DomainPoint tile_;
   };

   // Maps tasks in a 3D index space launch on ranks according to
   // the default tiling logic (see description above).
   class RankTiling3DFunctor : public SplinteringFunctor {
   public:
         RankTiling3DFunctor(Runtime* rt, SampleMapping& parent, bool fold)
            : SplinteringFunctor(rt, parent), fold_(fold) {}
   public:
      virtual ShardID shard(const DomainPoint& point,
                            const Domain& full_space,
                            const size_t total_shards) {
         assert(point.get_dim() == 3);
         unsigned x = fold_ ? point[0] % parent_.ranks_per_dim_[0] : point[0];
         unsigned y = fold_ ? point[1] % parent_.ranks_per_dim_[1] : point[1];
         unsigned z = fold_ ? point[2] % parent_.ranks_per_dim_[2] : point[2];
         CHECK(0 <= x && x < parent_.ranks_per_dim_[0] &&
               0 <= y && y < parent_.ranks_per_dim_[1] &&
               0 <= z && z < parent_.ranks_per_dim_[2],
               "Unexpected point on index space launch");
         return x * parent_.ranks_per_dim_[1]
                  * parent_.ranks_per_dim_[2] +
                y * parent_.ranks_per_dim_[2] +
                z;
      }

      virtual SplinterID splinter(const DomainPoint &point) {
         assert(point.get_dim() == 3);
         unsigned x = fold_ ? point[0] % parent_.ranks_per_dim_[0] : point[0];
         unsigned y = fold_ ? point[1] % parent_.ranks_per_dim_[1] : point[1];
         unsigned z = fold_ ? point[2] % parent_.ranks_per_dim_[2] : point[2];
         CHECK(0 <= x && x < parent_.ranks_per_dim_[0] &&
               0 <= y && y < parent_.ranks_per_dim_[1] &&
               0 <= z && z < parent_.ranks_per_dim_[2],
               "Unexpected point on index space launch");
         return x * parent_.tiles_per_rank_[1]
                  * parent_.tiles_per_rank_[2] +
                y * parent_.tiles_per_rank_[2] +
                z;
      }

   private:
      bool fold_;
   };

   // Maps tasks in a 2D index space launch, by extending each domain point to a
   // 3D tile and deferring to the default strategy.
   // Parameter `dim` controls which dimension to add.
   // Parameter `dir` controls which extreme of that dimension to set.
   class RankTiling2DFunctor : public SplinteringFunctor {
   public:
      RankTiling2DFunctor(Runtime* rt, SampleMapping& parent,
                      unsigned dim, bool dir)
         : SplinteringFunctor(rt, parent), dim_(dim), dir_(dir) {}

   public:
      virtual ShardID shard(const DomainPoint& point,
                            const Domain& full_space,
                            const size_t total_shards) {
         return parent_.rank_tiling_3d_functor_[0]->shard
                (to_point_3d(point), full_space, total_shards);
      }

      virtual SplinterID splinter(const DomainPoint &point) {
         return parent_.rank_tiling_3d_functor_[0]->splinter(to_point_3d(point));
      }

   private:
      DomainPoint to_point_3d(const DomainPoint& point) const {
         assert(point.get_dim() == 2);
         unsigned coord =
            (dim_ == 0) ? (dir_ ? 0 : parent_.ranks_per_dim_[0]-1) :
            (dim_ == 1) ? (dir_ ? 0 : parent_.ranks_per_dim_[1]-1) :
           /*dim_ == 2*/  (dir_ ? 0 : parent_.ranks_per_dim_[2]-1) ;
         return
            (dim_ == 0) ? Point<3>(coord, point[0], point[1]) :
            (dim_ == 1) ? Point<3>(point[0], coord, point[1]) :
           /*dim_ == 2*/  Point<3>(point[0], point[1], coord) ;
      }

   private:
      unsigned dim_;
      bool dir_;
   };

   // Maps every task to the same shard & splinter (the ones corresponding to
   // the rank specified in the constructor).
   class RankHardcodedFunctor : public SplinteringFunctor {
   public:
      RankHardcodedFunctor(Runtime* rt,
                       SampleMapping& parent,
                       const DomainPoint& tile)
         : SplinteringFunctor(rt, parent), tile_(tile) {}
   public:
      virtual ShardID shard(const DomainPoint& point,
                            const Domain& full_space,
                            const size_t total_shards) {
         return parent_.rank_tiling_3d_functor_[0]->shard(tile_, full_space, total_shards);
      }

      virtual SplinterID splinter(const DomainPoint &point) {
         return parent_.rank_tiling_3d_functor_[0]->splinter(tile_);
      }
   private:
      DomainPoint tile_;
   };

private:
   unsigned tiles_per_rank_[3];
   unsigned ranks_per_dim_[3];
   AddressSpace first_rank_;
   Tiling3DFunctor* tiling_3d_functor_;
   Tiling2DFunctor* tiling_2d_functors_[3][2];
   std::vector<HardcodedFunctor*> hardcoded_functors_;
   RankTiling3DFunctor* rank_tiling_3d_functor_[2];
   RankTiling2DFunctor* rank_tiling_2d_functors_[3][2];
   std::vector<RankHardcodedFunctor*> rank_hardcoded_functors_;
   // Samples passed using -lp (low-priority) are mapped differently (e.g., to CPUs)
   bool lowpriority_;
};

AddressSpace SplinteringFunctor::get_rank(const DomainPoint &point) {
   return parent_.get_rank(shard(point, Domain(), 0));
}

//=============================================================================
// MAPPER CLASS: CONSTRUCTOR
//=============================================================================

class PrometeoMapper : public DefaultMapper {
public:
   PrometeoMapper(Runtime* rt, Machine machine, Processor local)
      : DefaultMapper(rt->get_mapper_runtime(), machine, local, "prometeo_mapper"),
      all_procs_(remote_cpus.size()),
      all_next_io_proc_(remote_cpus.size()) {

      Processor::Kind kind = local_proc.kind();
      auto pid = local_proc.id;
      switch (kind) {
         // Latency-optimized cores (LOCs) are CPUs
         case Processor::LOC_PROC:
         {
            LOG.debug() << "  Processor ID " << std::hex << pid << " is CPU";
            break;
         }
         // Throughput-optimized cores (TOCs) are GPUs
         case Processor::TOC_PROC:
         {
            LOG.debug() << "  Processor ID " << std::hex << pid << " is GPU";
            break;
         }
         // Throughput-optimized cores (TOCs) are GPUs
         case Processor::OMP_PROC:
         {
            LOG.debug() << "  Processor ID " << std::hex <<  pid << " is OMP";
            break;
         }
         // Processor for doing I/O
         case Processor::IO_PROC:
         {
            LOG.debug() << "  Processor ID " << std::hex << pid << " is I/O Proc";
            break;
         }
         // Utility processors are helper processors for
         // running Legion runtime meta-level tasks and
         // should not be used for running application tasks
         case Processor::UTIL_PROC:
         {
            LOG.debug() << "  Processor ID " << std::hex << pid << " is utility";
            break;
         }
         default:
         {
            LOG.debug() << "  Processor ID " << std::hex << pid << " is have no clue:";
         }
      }

      // Set the umask of the process to clear S_IWGRP and S_IWOTH.
      umask(022);
      // Assign ranks sequentially to samples, each sample getting one rank for
      // each super-tile.
      auto process_config = [&](const Config& config, AddressSpace reqd_ranks) {
         CHECK(config.Mapping.tiles[0] > 0 &&
               config.Mapping.tiles[1] > 0 &&
               config.Mapping.tiles[2] > 0 &&
               config.Mapping.tilesPerRank[0] > 0 &&
               config.Mapping.tilesPerRank[1] > 0 &&
               config.Mapping.tilesPerRank[2] > 0 &&
               config.Mapping.tiles[0] % config.Mapping.tilesPerRank[0] == 0 &&
               config.Mapping.tiles[1] % config.Mapping.tilesPerRank[1] == 0 &&
               config.Mapping.tiles[2] % config.Mapping.tilesPerRank[2] == 0,
               "Invalid tiling for sample %lu", sample_mappings_.size() + 1);
         sample_mappings_.emplace_back(rt, config, reqd_ranks);
      };

      unsigned supplied_ranks = remote_cpus.size();
      AddressSpace reqd_ranks = 0;
      AddressSpace reqd_ranks_lp = 0;

      // Locate all config files specified on the command-line arguments.
      InputArgs args = Runtime::get_input_args();
      for (int i = 0; i < args.argc; ++i) {
         if (EQUALS(args.argv[i], "-i") && i < args.argc-1) {
            Config config;
            parse_Config(&config, args.argv[i+1]);
            process_config(config, reqd_ranks);
            reqd_ranks += sample_mappings_.back().num_ranks();
         } else if (EQUALS(args.argv[i], "-lp") && i < args.argc-1) {
            Config config;
            parse_Config(&config, args.argv[i+1]);
            process_config(config, reqd_ranks_lp);
            sample_mappings_.back().setLowpriority();
            auto snranks = sample_mappings_.back().num_ranks();
            LOG.debug() << std::hex << local_proc.id << "] -lp snranks: " << snranks << " supplied_ranks: " << supplied_ranks << " reqd_ranks_lp: " << reqd_ranks_lp;
            // Verify that we have enough ranks.
            CHECK(snranks <= supplied_ranks,
               "%u rank(s) required, but %u rank(s) supplied to Legion",
               snranks, supplied_ranks);
            // Just to make sure that there are enough ranks for the LP samples.
            // If not, just start from rank 0 for the next sample.
            if (snranks + reqd_ranks_lp >= supplied_ranks) {
               reqd_ranks_lp = 0;
            } else {
               reqd_ranks_lp += snranks;
            }
            LOG.debug() << std::hex << local_proc.id << "] -lp reqd_ranks_lp: " << reqd_ranks_lp;
         }
      }
      // Verify that we have enough ranks.
      CHECK(reqd_ranks <= supplied_ranks,
            "%u rank(s) required, but %u rank(s) supplied to Legion",
            reqd_ranks, supplied_ranks);
      if (reqd_ranks < supplied_ranks) {
         LOG.warning() << supplied_ranks << " rank(s) supplied to Legion,"
                       << " but only " << reqd_ranks << " required";
      }
      // Cache processor information.
      Machine::ProcessorQuery query(machine);
      for (auto it = query.begin(); it != query.end(); it++) {
         AddressSpace rank = it->address_space();
         Processor::Kind kind = it->kind();
         get_procs(rank, kind).push_back(*it);
      }
      // Initialize next_io_proc to use first IO proc
      for (auto it = query.begin(); it != query.end(); it++) {
         AddressSpace rank = it->address_space();
         all_next_io_proc_[rank] = 0;
      }
      // Verify machine configuration.
      for (AddressSpace rank = 0; rank < remote_cpus.size(); ++rank) {
         CHECK(get_procs(rank, Processor::IO_PROC).size() > 0,
               "No IO processor on rank %u", rank);
      }
   }

//=============================================================================
// MAPPER CLASS: MAPPING LOGIC
//=============================================================================

private:
   std::vector<unsigned> find_sample_ids(const MapperContext ctx,
                                         const Task& task) const {
      std::vector<unsigned> sample_ids;
      // Tasks called on regions: read the SAMPLE_ID_TAG from the region
      if (task.is_index_space ||
         EQUALS(task.get_task_name(), "DummyAverages") ||
         EQUALS(task.get_task_name(), "InitializeNodeGrid") ||
         EQUALS(task.get_task_name(), "ComputeRecycleAveragePosition") ||
         EQUALS(task.get_task_name(), "InitializeBoundarLayerData") ||
         EQUALS(task.get_task_name(), "GetRescalingData") ||
         EQUALS(task.get_task_name(), "cache_grid_translation") ||
#ifdef ELECTRIC_FIELD
         EQUALS(task.get_task_name(), "initCoefficients")  ||
         EQUALS(task.get_task_name(), "InitWaveNumbers") ||
#endif
         STARTS_WITH(task.get_task_name(), "FastInterp")) {

         CHECK(!task.regions.empty(),
               "Expected region argument in call to %s", task.get_task_name());
         const RegionRequirement& req = task.regions[0];
         LogicalRegion region = req.region.exists() ? req.region
            : runtime->get_parent_logical_region(ctx, req.partition);
         region = get_root(ctx, region);
         const void* info = NULL;
         size_t info_size = 0;
         bool success = runtime->retrieve_semantic_information
            (ctx, region, SAMPLE_ID_TAG, info, info_size,
             false/*can_fail*/, true/*wait_until_ready*/);
         CHECK(success, "Missing SAMPLE_ID_TAG semantic information on region");
         assert(info_size == sizeof(unsigned));
         sample_ids.push_back(*static_cast<const unsigned*>(info));
      }
      // Tasks with Config as 1st argument: read config.Mapping.sampleId
      else if (EQUALS(task.get_task_name(), "workSingle")) {
         const Config* config = static_cast<const Config*>(first_arg(task));
         sample_ids.push_back(static_cast<unsigned>(config->Mapping.sampleId));
      }
      // Helper & I/O tasks: go up one level to the work task
      else if (STARTS_WITH(task.get_task_name(), "Console_Write") ||
               STARTS_WITH(task.get_task_name(), "Probe_Write") ||
               EQUALS(task.get_task_name(), "createDir") ||
               EQUALS(task.get_task_name(), "dumpMasterFile") ||
               EQUALS(task.get_task_name(), "writeTileAttr") ||
               EQUALS(task.get_task_name(), "readTileAttr") ||
               EQUALS(task.get_task_name(), "__dummy") ||
               STARTS_WITH(task.get_task_name(), "__unary_") ||
               STARTS_WITH(task.get_task_name(), "__binary_")) {
         assert(task.parent_task != NULL);
         sample_ids = find_sample_ids(ctx, *(task.parent_task));
      }
      // Other tasks: fail and notify the user
      else {
         CHECK(false, "Unhandled task in find_sample_ids: %s",
               task.get_task_name());
      }
      // Sanity checks
      assert(!sample_ids.empty());
      for (unsigned sample_id : sample_ids) {
         assert(sample_id < sample_mappings_.size());
      }
      return sample_ids;
   }

   unsigned find_sample_id(const MapperContext ctx, const Task& task) const {
      return find_sample_ids(ctx, task)[0];
   }

   DomainPoint find_tile(const MapperContext ctx,
                         const Task& task) const {
      // Tasks that should run on the first rank of their sample's allocation
      if (EQUALS(task.get_task_name(), "workSingle") ||
          EQUALS(task.get_task_name(), "workDual") ||
          EQUALS(task.get_task_name(), "cache_grid_translation") ||
          STARTS_WITH(task.get_task_name(), "Console_Write") ||
          STARTS_WITH(task.get_task_name(), "Probe_Write") ||
          EQUALS(task.get_task_name(), "createDir") ||
          EQUALS(task.get_task_name(), "__dummy") ||
          EQUALS(task.get_task_name(), "DummyAverages") ||
          EQUALS(task.get_task_name(), "InitializeNodeGrid") ||
          EQUALS(task.get_task_name(), "ComputeRecycleAveragePosition") ||
          EQUALS(task.get_task_name(), "InitializeBoundarLayerData") ||
          EQUALS(task.get_task_name(), "GetRescalingData") ||
#ifdef ELECTRIC_FIELD
          EQUALS(task.get_task_name(), "initCoefficients")  ||
          EQUALS(task.get_task_name(), "InitWaveNumbers") ||
#endif
          EQUALS(task.get_task_name(), "dumpMasterFile") ||
          EQUALS(task.get_task_name(), "writeTileAttr") ||
          EQUALS(task.get_task_name(), "readTileAttr") ||
          STARTS_WITH(task.get_task_name(), "FastInterp") ||
          STARTS_WITH(task.get_task_name(), "__unary_") ||
          STARTS_WITH(task.get_task_name(), "__binary_")) {
         return Point<3>(0,0,0);
      }
      // Other tasks: fail and notify the user
      else {
         CHECK(false, "Unhandled task in find_tile: %s", task.get_task_name());
         return Point<3>(0,0,0);
      }
   }

   SplinteringFunctor* pick_functor(const MapperContext ctx,
                                    const Task& task) {
      // 3D index space tasks
      if (task.is_index_space && task.index_domain.get_dim() == 3) {
         unsigned sample_id = find_sample_id(ctx, task);
         SampleMapping& mapping = sample_mappings_[sample_id];
         // IO of 3D partitioned regions is managed by each rank
         if (STARTS_WITH(task.get_task_name(), "dumpTile") ||
             STARTS_WITH(task.get_task_name(), "loadTile")) {
            return mapping.rank_tiling_3d_functor(true);
         } else {
            return mapping.tiling_3d_functor();
         }
      }
      // 2D index space tasks
      else if (task.is_index_space && task.index_domain.get_dim() == 2) {
         unsigned sample_id = find_sample_id(ctx, task);
         SampleMapping& mapping = sample_mappings_[sample_id];
         // IO of 2D partitioned regions
         if (STARTS_WITH(task.get_task_name(), "dumpTile") ||
             STARTS_WITH(task.get_task_name(), "loadTile")) {
            return mapping.hardcoded_functor(Point<3>(0,0,0));
         } else {
            CHECK(false, "Unexpected 2D domain on index space launch of task %s",
                  task.get_task_name());
            return NULL;
         }
      }
      // 1D index space tasks
      else if (task.is_index_space && task.index_domain.get_dim() == 1) {
         unsigned sample_id = find_sample_id(ctx, task);
         SampleMapping& mapping = sample_mappings_[sample_id];
         // IO of 1D partitioned regions
         if (STARTS_WITH(task.get_task_name(), "dumpTile") ||
             STARTS_WITH(task.get_task_name(), "loadTile")) {
            return mapping.hardcoded_functor(Point<3>(0,0,0));
         } else {
            CHECK(false, "Unexpected 1D domain on index space launch of task %s",
                  task.get_task_name());
            return NULL;
         }
      }
      // Sample-specific tasks that are launched individually on each tile
      else if (EQUALS(task.get_task_name(), "workSingle") ||
               EQUALS(task.get_task_name(), "workDual") ||
               EQUALS(task.get_task_name(), "DummyAverages") ||
               EQUALS(task.get_task_name(), "InitializeNodeGrid") ||
               EQUALS(task.get_task_name(), "ComputeRecycleAveragePosition") ||
               EQUALS(task.get_task_name(), "InitializeBoundarLayerData") ||
               EQUALS(task.get_task_name(), "GetRescalingData") ||
#ifdef ELECTRIC_FIELD
               EQUALS(task.get_task_name(), "initCoefficients")  ||
               EQUALS(task.get_task_name(), "InitWaveNumbers") ||
#endif
               EQUALS(task.get_task_name(), "dumpMasterFile") ||
               EQUALS(task.get_task_name(), "writeTileAttr") ||
               EQUALS(task.get_task_name(), "readTileAttr") ||
               EQUALS(task.get_task_name(), "cache_grid_translation") ||
               STARTS_WITH(task.get_task_name(), "FastInterp") ||
               STARTS_WITH(task.get_task_name(), "Console_Write") ||
               STARTS_WITH(task.get_task_name(), "Probe_Write") ||
               EQUALS(task.get_task_name(), "createDir") ||
               EQUALS(task.get_task_name(), "__dummy") ||
               STARTS_WITH(task.get_task_name(), "__unary_") ||
               STARTS_WITH(task.get_task_name(), "__binary_")) {
         unsigned sample_id = find_sample_id(ctx, task);
         SampleMapping& mapping = sample_mappings_[sample_id];
         DomainPoint tile = find_tile(ctx, task);
         return mapping.hardcoded_functor(tile);
      }
      // Other tasks: fail and notify the user
      else {
         CHECK(false, "Unhandled task in pick_functor: %s", task.get_task_name());
         return NULL;
      }
   }

//=============================================================================
// MAPPER CLASS: MAJOR OVERRIDES
//=============================================================================

public:
   // Control-replicate work tasks.
   virtual void select_task_options(const MapperContext ctx,
                                    const Task& task,
                                   TaskOptions& output) {
      DefaultMapper::select_task_options(ctx, task, output);
      output.replicate =
         EQUALS(task.get_task_name(), "workSingle") ||
         EQUALS(task.get_task_name(), "workDual");
   }

   virtual void default_policy_rank_processor_kinds(MapperContext ctx,
                                                    const Task& task,
                                                    std::vector<Processor::Kind>& ranking) {

      // the main task is deferred to the default mapping policy
      if (EQUALS(task.get_task_name(), "main")) {
         DefaultMapper::default_policy_rank_processor_kinds(ctx, task, ranking);

      // Work tasks: map to CPU processors
      } else if (EQUALS(task.get_task_name(), "workSingle") ||
                 EQUALS(task.get_task_name(), "workDual")) {
         ranking.push_back(Processor::LOC_PROC);

      // HDF5 tasks: map to IO processors
      } else if (STARTS_WITH(task.get_task_name(), "dumpTile")  ||
                 STARTS_WITH(task.get_task_name(), "loadTile")  ||
                 EQUALS(task.get_task_name(), "dumpMasterFile") ||
                 EQUALS(task.get_task_name(), "writeTileAttr")  ||
                 EQUALS(task.get_task_name(), "readTileAttr")) {
         ranking.push_back(Processor::IO_PROC);

      // Console tasks: map to IO processors
      } else if (STARTS_WITH(task.get_task_name(), "Console_Write") ||
                 EQUALS(     task.get_task_name(), "createDir")) {
         ranking.push_back(Processor::IO_PROC);

      // Probe output tasks: map to IO processors
      } else if (STARTS_WITH(task.get_task_name(), "Probe_Write")) {
         ranking.push_back(Processor::IO_PROC);

      // Other tasks: differ mapping depending whether it is Low priority or High priority
      } else {
         unsigned sample_id = find_sample_id(ctx, task);
         const SampleMapping& mapping = sample_mappings_[sample_id];

         // Restrict low fidelities to CPUs
         if (mapping.isLowpriority()) {
             ranking.resize(2);
             ranking[0] = Processor::OMP_PROC;
             ranking[1] = Processor::LOC_PROC;

         // Other tasks: defer to the default mapping policy
         } else {
             DefaultMapper::default_policy_rank_processor_kinds(ctx, task, ranking);
         }
      }
   }

#ifndef NO_LEGION_CONTROL_REPLICATION
   // Replicate each work task over all ranks assigned to the corresponding
   // sample(s).
   virtual void map_replicate_task(const MapperContext ctx,
                                   const Task& task,
                                   const MapTaskInput& input,
                                   const MapTaskOutput& default_output,
                                   MapReplicateTaskOutput& output) {
      // Read configuration.
      assert(!runtime->is_MPI_interop_configured(ctx));
      assert(EQUALS(task.get_task_name(), "workSingle") ||
             EQUALS(task.get_task_name(), "workDual"));
      VariantInfo info =
         default_find_preferred_variant(task, ctx, false/*needs_tight_bound*/);
      CHECK(task.regions.empty() && info.is_replicable,
            "Unexpected features on work task");
      std::vector<unsigned> sample_ids = find_sample_ids(ctx, task);
      // Create a replicant on the first CPU processor of each sample's ranks.
      for (unsigned sample_id : sample_ids) {
         const SampleMapping& mapping = sample_mappings_[sample_id];
         for (ShardID shard_id = 0; shard_id < mapping.num_ranks(); ++shard_id) {
            AddressSpace rank = mapping.get_rank(shard_id);
            Processor target_proc = get_procs(rank, info.proc_kind)[0];
            output.task_mappings.push_back(default_output);
            output.task_mappings.back().chosen_variant = info.variant;
            output.task_mappings.back().target_procs.push_back(target_proc);
            output.control_replication_map.push_back(target_proc);
         }
      }
   }
#endif

   // NOTE: Will only run if Legion is compiled with dynamic control replication.
   virtual void select_sharding_functor(const MapperContext ctx,
                                        const Task& task,
                                        const SelectShardingFunctorInput& input,
                                        SelectShardingFunctorOutput& output) {
      output.chosen_functor = pick_functor(ctx, task)->id;
   }

   virtual Processor default_policy_select_initial_processor(MapperContext ctx,
                                                             const Task& task) {
      // Index space tasks: defer to the default mapping policy; slice_task will
      // eventually be called to do the mapping properly
      if (task.is_index_space) {
         return DefaultMapper::default_policy_select_initial_processor(ctx, task);
      }
      // Main task: defer to the default mapping policy
      else if (EQUALS(task.get_task_name(), "main")) {
         return DefaultMapper::default_policy_select_initial_processor(ctx, task);
      }
      // Other tasks
      else {
         unsigned sample_id = find_sample_id(ctx, task);
         DomainPoint tile = find_tile(ctx, task);
         VariantInfo info = default_find_preferred_variant(task, ctx, false/*needs_tight_bound*/);
         SplinteringFunctor* functor = pick_functor(ctx, task);
         Processor target_proc = select_proc(tile, info.proc_kind, functor);
         LOG.debug() << "Sample " << sample_id
                     << ": Task " << task.get_task_name()
                     << ": Sequential launch"
                     << ": Tile " << tile
                     << ": Processor " << target_proc;
         return target_proc;
      }
   }

   virtual void slice_task(const MapperContext ctx,
                           const Task& task,
                           const SliceTaskInput& input,
                           SliceTaskOutput& output) {
      output.verify_correctness = false;
      unsigned sample_id = find_sample_id(ctx, task);
      VariantInfo info = default_find_preferred_variant(task, ctx, false/*needs_tight_bound*/);
      if (sample_mappings_[sample_id].isLowpriority() and
          (info.proc_kind != Processor::IO_PROC) ) {
         // Low-priority stuff that does not need an IO_PROC gets mapped either on
         // - OMP_PROC
         // - LOC_PROC
         Processor::Kind lpkind = Processor::LOC_PROC;
#ifdef REALM_USE_OPENMP
         std::vector<VariantID> variants;
         runtime->find_valid_variants(ctx, task.task_id, variants, Processor::OMP_PROC);
         if ( ! variants.empty() ) {
            lpkind = Processor::OMP_PROC;
         }
#endif
         info = default_find_preferred_variant(task, ctx,
                                               false/*needs_tight_bound*/,
                                               true/*cache*/,
                                               lpkind);
      }
      SplinteringFunctor* functor = pick_functor(ctx, task);
      for (Domain::DomainPointIterator it(input.domain); it; it++) {
         Processor target_proc = select_proc(it.p, info.proc_kind, functor);
         output.slices.emplace_back(Domain(it.p, it.p), target_proc,
                                    false/*recurse*/,
                                    (target_proc.kind() == Processor::IO_PROC) /*stealable*/);
         LOG.debug() << "Sample " << sample_id
                     << ": Task " << task.get_task_name()
                     << ": Index space launch"
                     << ": Tile " << it.p
                     << ": Processor " << target_proc;
      }
   }

   virtual TaskPriority default_policy_select_task_priority(MapperContext ctx,
                                                            const Task& task) {
      // Unless handled specially below, all tasks have the same priority.
      int priority = 0;

      if ( ! EQUALS(task.get_task_name(), "main") ) {
         unsigned sample_id = find_sample_id(ctx, task);
         const SampleMapping& mapping = sample_mappings_[sample_id];

         if (mapping.isLowpriority()) {
            priority = -1;
         } else {
            if (STARTS_WITH(task.get_task_name(), "UpdateShockSensor") ||
                STARTS_WITH(task.get_task_name(), "UpdateUsing") ||
                STARTS_WITH(task.get_task_name(), "UpdateVars") ||
                STARTS_WITH(task.get_task_name(), "UpdateChemistry") ||
                STARTS_WITH(task.get_task_name(), "AddChemistrySources") ||
                STARTS_WITH(task.get_task_name(), "AddBodyForces") ||
                STARTS_WITH(task.get_task_name(), "AddIonWindSources") ||
                STARTS_WITH(task.get_task_name(), "AddLaser") ||
                STARTS_WITH(task.get_task_name(), "workSingle")) {
               priority = 1;
            }
         }
      }
      return priority;
   }

   // Send each cross-section explicit copy to the first rank of the first
   // section, to be mapped further.
   // NOTE: Will only run if Legion is compiled with dynamic control replication.
   virtual void select_sharding_functor(const MapperContext ctx,
                                        const Copy& copy,
                                        const SelectShardingFunctorInput& input,
                                        SelectShardingFunctorOutput& output) {
      CHECK(copy.parent_task != NULL &&
            EQUALS(copy.parent_task->get_task_name(), "workDual"),
            "Unsupported: Sharded copy outside of workDual");
      unsigned sample_id = find_sample_id(ctx, *(copy.parent_task));
      SampleMapping& mapping = sample_mappings_[sample_id];
      output.chosen_functor = mapping.hardcoded_functor(Point<3>(0,0,0))->id;
   }

   virtual void map_copy(const MapperContext ctx,
                         const Copy& copy,
                         const MapCopyInput& input,
                         MapCopyOutput& output) {
      // For HDF copies, defer to the default mapping policy.
      if (EQUALS(copy.parent_task->get_task_name(), "dumpTile") ||
          EQUALS(copy.parent_task->get_task_name(), "loadTile")) {
         DefaultMapper::map_copy(ctx, copy, input, output);
         return;
      }
      CHECK(false, "Unsupported: map_copy");
   }

   virtual void select_sharding_functor(const MapperContext ctx,
                                        const Fill& fill,
                                        const SelectShardingFunctorInput& input,
                                        SelectShardingFunctorOutput& output) {
		CHECK(fill.parent_task != NULL,
            "Unsupported: Sharded Fill does not have parent partition");
      unsigned sample_id = find_sample_id(ctx, *(fill.parent_task));
      if (fill.is_index_space && fill.index_domain.get_dim() == 3) {
         SampleMapping& mapping = sample_mappings_[sample_id];
         if (fill.index_domain.get_volume() == mapping.num_tiles()) {
            output.chosen_functor = mapping.tiling_3d_functor()->id;
            LOG.debug() << "Sample " << sample_id
                        << ": Fill from parent task " << fill.parent_task->get_task_name()
                        << ": sharded using tiling_3d_functor";
         } else {
            output.chosen_functor = mapping.rank_tiling_3d_functor(true)->id;
            LOG.debug() << "Sample " << sample_id
                        << ": Fill from parent task " << fill.parent_task->get_task_name()
                        << ": sharded using rank_tiling_3d_functor";
         }
      } else {
         output.chosen_functor = pick_functor(ctx, *(fill.parent_task))->id;
         LOG.debug() << "Sample " << sample_id
                     << ": Fill from parent task " << fill.parent_task->get_task_name()
                     << ": sharded using pick_functor";
      }
   }

   // NOTE: Will only run if Legion is compiled with dynamic control replication.
   // Send each dependent partition operation to the rank corresponding
   // to its tile (3d index_space launched) or to the rank of its parent task.
   virtual void select_sharding_functor(const MapperContext ctx,
                                        const Partition& partition,
                                        const SelectShardingFunctorInput& input,
                                        SelectShardingFunctorOutput& output) {
      CHECK(partition.parent_task != NULL &&
           (EQUALS(partition.parent_task->get_task_name(), "workSingle") ||
            EQUALS(partition.parent_task->get_task_name(), "workDual")),
            "Unsupported: Sharded partition outside of workSingle or workDual");
      unsigned sample_id = find_sample_id(ctx, *(partition.parent_task));
      SampleMapping& mapping = sample_mappings_[sample_id];
//      const char *name = get_partition_name(ctx, partition.requirement.partition);
//      CHECK(name != NULL, "Found an unnamed partition");

      // 3D index space tasks
      if (partition.is_index_space && partition.index_domain.get_dim() == 3) {
         if (partition.index_domain.get_volume() == mapping.num_tiles()) {
            output.chosen_functor = mapping.tiling_3d_functor()->id;
            LOG.debug() << "Sample " << sample_id
                        << ": Partition parent task " << partition.parent_task->get_task_name()
//                        << ": Partition parent partition " << name
                        << ": sharded using tiling_3d_functor";
         } else {
            output.chosen_functor = mapping.rank_tiling_3d_functor(true)->id;
            LOG.debug() << "Sample " << sample_id
                        << ": Partition parent task " << partition.parent_task->get_task_name()
//                        << ": Partition parent partition " << name
                        << ": sharded using rank_tiling_3d_functor";
         }
      } else {
         LOG.debug() << "Sample " << sample_id
                     << ": Partition parent task " << partition.parent_task->get_task_name()
//                     << ": Partition parent partition " << name
                     << ": sharded using pick_functor";
         output.chosen_functor = pick_functor(ctx, *(partition.parent_task))->id;
      }
   }

   //--------------------------------------------------------------------------
   // Work stealing algorithm
   // For now we allow only IO_PROCs to steal work from the other IO_PROCs of
   // the same shard
   //--------------------------------------------------------------------------

   virtual void select_steal_targets(const MapperContext         ctx,
                                     const SelectStealingInput&  input,
                                           SelectStealingOutput& output) {
      LOG.spew("select_steal_targets in %s", get_mapper_name());
      output.targets.clear();
      if (local_kind == Processor::IO_PROC) {
         // Add all the IO_PROCs of my rank as potential targets
         const std::vector<Processor>& procs = get_procs(node_id, local_kind);
         for (auto p : procs) {
            if (local_proc == p) continue;
            output.targets.insert(p);
            LOG.debug() << "Processor " << local_proc
                        << ": Adding Processor " << p
                        << " as a steal targets";
         }
      }
   }

   virtual void permit_steal_request(const MapperContext       ctx,
                                     const StealRequestInput&  input,
                                           StealRequestOutput& output) {
      LOG.spew("permit_steal_request in %s", get_mapper_name());
      output.stolen_tasks.clear();
      // Only Processor::IO_PROC are allowed to steal work for now
      assert(input.thief_proc.kind() == Processor::IO_PROC);
      // Find rank of the stealing proc
      const AddressSpace thief_rank = get_proc_rank(input.thief_proc);
      // Iterate over stealable tasks
      for (auto task : input.stealable_tasks) {
         if ((task->current_proc.kind() == input.thief_proc.kind()) and
             (get_proc_rank(task->current_proc) == thief_rank)) {
            // The task was assigned to a proc of the same kind in the same rank
            // let's steal it
            output.stolen_tasks.insert(task);
            unsigned sample_id = find_sample_id(ctx, *task);
            LOG.debug() << "Sample " << sample_id
                        << ": Processor " << input.thief_proc
                        << " is stealing the Task " << task->get_task_name()
                        << " from Processor " << task->current_proc;
         }
      }
   }

//=============================================================================
// MAPPER CLASS: MINOR OVERRIDES
//=============================================================================

public:
   // TODO: Select appropriate memories for instances that will be communicated,
   // (e.g. parallelizer-created ghost partitions), such as RDMA memory,
   // zero-copy memory.
   virtual Memory default_policy_select_target_memory(MapperContext ctx,
                                                      Processor target_proc,
                                                      const RegionRequirement& req,
                                                      MemoryConstraint mc) {
#ifdef ELECTRIC_FIELD
      // A root region uses the default policy
      if (!runtime->has_parent_logical_partition(ctx, req.region)) {
         LOG.debug() << "Root region uses default target memory";
         return DefaultMapper::default_policy_select_target_memory(ctx, target_proc, req, mc);
      }

      // Get partition name
      LogicalPartition parent_partition = runtime->get_parent_logical_partition(ctx, req.region);
      const char *name = get_partition_name(ctx, parent_partition);
      CHECK(name != NULL, "Found an unnamed partition");

#ifdef REALM_USE_CUDA
      if (EQUALS(name, "Poisson_plans"))
         // Put FFT plans in zero-copy memory.
         return Utilities::MachineQueryInterface::find_memory_kind(machine, target_proc,
                                                                   Memory::Z_COPY_MEM);
#endif
#endif

      // Otherwise go through the standard path
      return DefaultMapper::default_policy_select_target_memory(ctx, target_proc, req, mc);
   }

   // Disable an optimization done by the default mapper (attempts to reuse an
   // instance that covers a superset of the requested index space, by searching
   // higher up the partition tree).
   virtual LogicalRegion default_policy_select_instance_region(MapperContext ctx,
                                                               Memory target_memory,
                                                               const RegionRequirement& req,
                                                               const LayoutConstraintSet& constraints,
                                                               bool force_new_instances,
                                                               bool meets_constraints) {

      // A root region does not need any special care
      if (!runtime->has_parent_logical_partition(ctx, req.region)) {
         LOG.debug() << "Root region assigned to its own instance";
         return req.region;
      }

      LogicalPartition parent_partition = runtime->get_parent_logical_partition(ctx, req.region);
      const char *name = get_partition_name(ctx, parent_partition);
      CHECK(name != NULL, "Found an unnamed partition");

      if (EQUALS(name, "p_All") ||
          EQUALS(name, "p_Interior") ||
          EQUALS(name, "p_AllBCs") ||
          EQUALS(name, "p_solved") ||
          EQUALS(name, "p_GradientGhosts") ||
          EQUALS(name, "p_AvgGradientGhosts") ||
          EQUALS(name, "p_MetricGhosts") ||
          EQUALS(name, "p_x_divg")  || EQUALS(name, "p_y_divg")  || EQUALS(name, "p_z_divg") ||
          EQUALS(name, "p_x_faces") || EQUALS(name, "p_y_faces") || EQUALS(name, "p_z_faces") ||
          STARTS_WITH(name, "p_xNeg") || STARTS_WITH(name, "p_xPos") ||
          STARTS_WITH(name, "p_yNeg") || STARTS_WITH(name, "p_yPos") ||
          STARTS_WITH(name, "p_zNeg") || STARTS_WITH(name, "p_zPos") ||
          EQUALS(name, "p_XFluxGhosts")    || EQUALS(name, "p_YFluxGhosts")    || EQUALS(name, "p_ZFluxGhosts")    ||
          EQUALS(name, "p_XDiffGhosts")    || EQUALS(name, "p_YDiffGhosts")    || EQUALS(name, "p_ZDiffGhosts")    ||
          EQUALS(name, "p_XDiffGradGhosts")|| EQUALS(name, "p_YDiffGradGhosts")|| EQUALS(name, "p_ZDiffGradGhosts")||
          EQUALS(name, "p_XEulerGhosts")   || EQUALS(name, "p_YEulerGhosts")   || EQUALS(name, "p_ZEulerGhosts")   ||
          EQUALS(name, "p_XSensorGhosts2") || EQUALS(name, "p_YSensorGhosts2") || EQUALS(name, "p_ZSensorGhosts2") ||
          EQUALS(name, "p_XSensorGhosts")  || EQUALS(name, "p_YSensorGhosts")  || EQUALS(name, "p_ZSensorGhosts")  ||
          EQUALS(name, "p_Fluid_YZAvg")    || EQUALS(name, "p_Fluid_XZAvg")    || EQUALS(name, "p_Fluid_XYAvg")    ||
          EQUALS(name, "p_Fluid_XAvg")     || EQUALS(name, "p_Fluid_YAvg")     || EQUALS(name, "p_Fluid_ZAvg")     ||
          EQUALS(name, "BCPlane")          || EQUALS(name, "p_Laser")) {

         DomainPoint tile = runtime->get_logical_region_color_point(ctx, req.region);
         LogicalRegion root_region = get_root(ctx, req.region);
         LogicalPartition primary_partition = get_partition_by_name(ctx, root_region, "p_AllWithGhosts");
         if (primary_partition == LogicalPartition::NO_PART) {
            // If p_AllWithGhosts has not been created yet use p_All
            LOG.debug() << "Region of " << name
                        << ": Tile " << tile
                        << " is mapped on corresponding instance of p_All";
            LogicalPartition p_All = get_partition_by_name(ctx, root_region, "p_All");
            assert(p_All != LogicalPartition::NO_PART);
            return runtime->get_logical_subregion_by_color(ctx, p_All, tile);
         } else {
            // otherwise map everything on p_AllWithGhosts
            LOG.debug() << "Region of " << name
                        << ": Tile " << tile
                        << " is mapped on corresponding instance of p_AllWithGhosts";
            assert(primary_partition != LogicalPartition::NO_PART);
            return runtime->get_logical_subregion_by_color(ctx, primary_partition, tile);
         }
      }

      LOG.debug() << "Region of " << name << " is mapped on its own instance";
      return req.region;
   }

   //--------------------------------------------------------------------------
   virtual void default_policy_select_sources(MapperContext ctx,
                                  const PhysicalInstance &target,
                                  const std::vector<PhysicalInstance> &sources,
                                  std::deque<PhysicalInstance> &ranking)
   //--------------------------------------------------------------------------
   {
      // Let the default mapper sort the sources by bandwidth
      DefaultMapper::default_policy_select_sources(ctx, target, sources, ranking);

//      std::ostringstream srank1;
//      for (std::deque<PhysicalInstance>::const_iterator it = ranking.begin();
//           it != ranking.end(); it++)
//      {
//         srank1 << *it << " ";
//      }
//      LOG.debug() << "Default rank for " << target << ": " << srank1.str().c_str();

      // Give priority to those with better overlapping
      std::vector<std::pair<PhysicalInstance,unsigned/*size of intersection*/>>
        cover_ranking(sources.size());

      Domain target_domain = target.get_instance_domain();
      for (std::deque<PhysicalInstance>::const_reverse_iterator it = ranking.rbegin();
           it != ranking.rend(); it++)
      {
        const unsigned idx = it - ranking.rbegin();
        const PhysicalInstance &source = (*it);
        Domain source_domain = source.get_instance_domain();
        Domain intersection = source_domain.intersection(target_domain);
        cover_ranking[idx] = std::pair<PhysicalInstance,unsigned>(source,intersection.get_volume());
      }

      // Sort them by the size of intersecting area
      std::stable_sort(cover_ranking.begin(), cover_ranking.end(), physical_sort_func);

      // Iterate from largest intersection, bandwidth to smallest
      ranking.clear();
      for (std::vector<std::pair<PhysicalInstance,unsigned>>::
            const_reverse_iterator it = cover_ranking.rbegin();
            it != cover_ranking.rend(); it++)
        ranking.push_back(it->first);

//      std::ostringstream srank2;
//      for (std::deque<PhysicalInstance>::const_iterator it = ranking.begin();
//           it != ranking.end(); it++)
//      {
//         srank2 << *it << " ";
//      }
//      LOG.debug() << "New rank for " << target << ": " << srank1.str().c_str();

   }

   // Disable an optimization done by the default mapper (extends the set of
   // eligible processors to include all the processors of the same type on the
   // target node).
   virtual void default_policy_select_target_processors(MapperContext ctx,
                                                        const Task &task,
                                                        std::vector<Processor> &target_procs) {
      target_procs.push_back(task.target_proc);
   }

   // Enable tracing.
   virtual void memoize_operation(const MapperContext ctx,
                                  const Mappable& mappable,
                                  const MemoizeInput& input,
                                  MemoizeOutput& output) {
      output.memoize = true;
   }

   // Shouldn't have to shard any of the following operations.
   virtual void select_sharding_functor(const MapperContext ctx,
                                        const Close& close,
                                        const SelectShardingFunctorInput& input,
                                        SelectShardingFunctorOutput& output) {
      CHECK(false, "Unsupported: Sharded Close");
   }
   virtual void select_sharding_functor(const MapperContext ctx,
                                        const Acquire& acquire,
                                        const SelectShardingFunctorInput& input,
                                        SelectShardingFunctorOutput& output) {
      CHECK(false, "Unsupported: Sharded Acquire");
   }
   virtual void select_sharding_functor(const MapperContext ctx,
                                        const Release& release,
                                        const SelectShardingFunctorInput& input,
                                        SelectShardingFunctorOutput& output) {
      CHECK(false, "Unsupported: Sharded Release");
   }
   virtual void select_sharding_functor(const MapperContext ctx,
                                        const MustEpoch& epoch,
                                        const SelectShardingFunctorInput& input,
                                        MustEpochShardingFunctorOutput& output) {
      CHECK(false, "Unsupported: Sharded MustEpoch");
   }

//=============================================================================
// MAPPER CLASS: HELPER METHODS
//=============================================================================

private:
   // This function selects the processor for a given tile and it has
   // two different behaviours depending on the processor kind:
   // - kind == IO_PROC: use a round-robin approach to assign task mapped on a
   //                    shard to different IO_PROC
   // - every other kind: assign the proc corresponding to the tile with the
   //                     approach described at the beginning of the file
   // NOTE: This function doesn't sanity check its input.
   Processor select_proc(const DomainPoint& tile,
                         const Processor::Kind kind,
                         SplinteringFunctor* functor) {
      const AddressSpace rank = functor->get_rank(tile);
      const std::vector<Processor>& procs = get_procs(rank, kind);
      Processor p;
      if (kind == Processor::IO_PROC) {
         // apply round robin
         if (all_next_io_proc_[rank] == procs.size())
            all_next_io_proc_[rank] = 0;
         p = procs[all_next_io_proc_[rank]++];
      } else {
         // Assign based on tile
         SplinterID splinter_id = functor->splinter(tile);
         p = procs[splinter_id % procs.size()];
      }
      return p;
   }

   std::vector<Processor>& get_procs(const AddressSpace rank, const Processor::Kind kind) {
      assert(rank < all_procs_.size());
      auto& rank_procs = all_procs_[rank];
      if (kind >= rank_procs.size()) {
         rank_procs.resize(kind + 1);
      }
      return rank_procs[kind];
   }

   AddressSpace get_proc_rank(const Processor & p) {
      AddressSpace r = all_procs_.size();
      for (AddressSpace rank = 0; rank < all_procs_.size(); ++rank) {
         const std::vector<Processor>& procs = get_procs(rank, p.kind());
         for (auto pr : procs)
            if (pr == p) r = rank;
      }
      assert(r < all_procs_.size());
      return r;
   }

   LogicalRegion get_root(const MapperContext ctx, LogicalRegion region) const {
      while (runtime->has_parent_logical_partition(ctx, region)) {
         region = runtime->get_parent_logical_region(ctx,
            runtime->get_parent_logical_partition(ctx, region));
      }
      return region;
   }

   const char* get_partition_name(MapperContext ctx,
                                  const LogicalPartition &lp) {
      const void *name = NULL;
      size_t size = 0;
      runtime->retrieve_semantic_information(ctx, lp, NAME_SEMANTIC_TAG, name, size, true, false);
      return (const char*)name;
   }

   LogicalPartition get_partition_by_name(MapperContext ctx,
                                          const LogicalRegion &region,
                                          const char *name) {
      std::set<Color> colors;
      runtime->get_index_space_partition_colors(ctx, region.get_index_space(), colors);
      for (std::set<Color>::const_iterator it = colors.begin(); it != colors.end(); ++it) {
         LogicalPartition lp = runtime->get_logical_partition_by_color(ctx, region, *it);
         const char *n = get_partition_name(ctx, lp);
         if (n != NULL && EQUALS(n, name)) return lp;
      }
      return LogicalPartition::NO_PART;
   }

//=============================================================================
// MAPPER CLASS: MEMBER VARIABLES
//=============================================================================

private:
   std::deque<SampleMapping> sample_mappings_;
   std::vector<std::vector<std::vector<Processor> > > all_procs_;
   std::vector< unsigned > all_next_io_proc_;
};

//=============================================================================
// MAPPER REGISTRATION
//=============================================================================

static void create_mappers(Machine machine,
                           Runtime* rt,
                           const std::set<Processor>& local_procs) {
   for (Processor proc : local_procs) {
      rt->replace_default_mapper(new PrometeoMapper(rt, machine, proc), proc);
   }
}

void register_mappers() {
   Runtime::add_registration_callback(create_mappers);
}
