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
                           new Tiling2DFunctor(rt, *this, 2, true )}} {
      for (unsigned x = 0; x < x_tiles(); ++x) {
         for (unsigned y = 0; y < y_tiles(); ++y) {
            for (unsigned z = 0; z < z_tiles(); ++z) {
               hardcoded_functors_.push_back(new HardcodedFunctor(rt, *this, Point<3>(x,y,z)));
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

private:
   unsigned tiles_per_rank_[3];
   unsigned ranks_per_dim_[3];
   AddressSpace first_rank_;
   Tiling3DFunctor* tiling_3d_functor_;
   Tiling2DFunctor* tiling_2d_functors_[3][2];
   std::vector<HardcodedFunctor*> hardcoded_functors_;
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
      all_procs_(remote_cpus.size()) {

      // Set the umask of the process to clear S_IWGRP and S_IWOTH.
      umask(022);
      // Assign ranks sequentially to samples, each sample getting one rank for
      // each super-tile.
      AddressSpace reqd_ranks = 0;
      auto process_config = [&](const Config& config) {
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
      // Locate all config files specified on the command-line arguments.
      InputArgs args = Runtime::get_input_args();
      for (int i = 0; i < args.argc; ++i) {
         if (EQUALS(args.argv[i], "-i") && i < args.argc-1) {
            Config config;
            parse_Config(&config, args.argv[i+1]);
            process_config(config);
            reqd_ranks += sample_mappings_.back().num_ranks();
         }
      }
      // Verify that we have enough ranks.
      unsigned supplied_ranks = remote_cpus.size();
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
         EQUALS(task.get_task_name(), "Exports.DummyAverages") ||
         EQUALS(task.get_task_name(), "ReduceAverages") ||
         EQUALS(task.get_task_name(), "cache_grid_translation") ||
         STARTS_WITH(task.get_task_name(), "readTileAttr")) {

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
      else if (STARTS_WITH(task.get_task_name(), "Exports.Console_Write") ||
               STARTS_WITH(task.get_task_name(), "Exports.Probe_Write") ||
               EQUALS(task.get_task_name(), "Exports.createDir") ||
               EQUALS(task.get_task_name(), "__dummy") ||
               STARTS_WITH(task.get_task_name(), "__unary_") ||
               STARTS_WITH(task.get_task_name(), "__binary_") ||
               STARTS_WITH(task.get_task_name(), "AffineTransform")) {
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
      // 3D index space tasks that are launched individually
      if (STARTS_WITH(task.get_task_name(), "readTileAttr")) {
         assert(!task.regions.empty() && task.regions[0].region.exists());
         DomainPoint tile =
            runtime->get_logical_region_color_point(ctx, task.regions[0].region);
         return tile;
      }
      // Tasks that should run on the first rank of their sample's allocation
      else if (EQUALS(task.get_task_name(), "workSingle") ||
               EQUALS(task.get_task_name(), "workDual") ||
               EQUALS(task.get_task_name(), "cache_grid_translation") ||
               STARTS_WITH(task.get_task_name(), "Exports.Console_Write") ||
               STARTS_WITH(task.get_task_name(), "Exports.Probe_Write") ||
               EQUALS(task.get_task_name(), "Exports.createDir") ||
               EQUALS(task.get_task_name(), "__dummy") ||
               EQUALS(task.get_task_name(), "Exports.DummyAverages") ||
               EQUALS(task.get_task_name(), "ReduceAverages") ||
               STARTS_WITH(task.get_task_name(), "__unary_") ||
               STARTS_WITH(task.get_task_name(), "__binary_") ||
               STARTS_WITH(task.get_task_name(), "AffineTransform")) {
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
         return mapping.tiling_3d_functor();
      }
      // 2D index space tasks
      else if (task.is_index_space && task.index_domain.get_dim() == 2) {
         unsigned sample_id = find_sample_id(ctx, task);
         SampleMapping& mapping = sample_mappings_[sample_id];
         // IO of 2D partitioned regions
         if (STARTS_WITH(task.get_task_name(), "dumpTile") ||
             STARTS_WITH(task.get_task_name(), "loadTile") ||
             STARTS_WITH(task.get_task_name(), "writeTileAttr")) {
            return mapping.hardcoded_functor(Point<3>(0,0,0));
         } else {
            CHECK(false, "Unexpected 2D domain on index space launch of task %s",
                  task.get_task_name());
            return NULL;
         }
      }
      // Sample-specific tasks that are launched individually
      else if (EQUALS(task.get_task_name(), "workSingle") ||
               EQUALS(task.get_task_name(), "workDual") ||
               EQUALS(task.get_task_name(), "Exports.DummyAverages") ||
               EQUALS(task.get_task_name(), "ReduceAverages") ||
               EQUALS(task.get_task_name(), "cache_grid_translation") ||
               STARTS_WITH(task.get_task_name(), "Exports.Console_Write") ||
               STARTS_WITH(task.get_task_name(), "Exports.Probe_Write") ||
               EQUALS(task.get_task_name(), "Exports.createDir") ||
               EQUALS(task.get_task_name(), "__dummy") ||
               STARTS_WITH(task.get_task_name(), "__unary_") ||
               STARTS_WITH(task.get_task_name(), "__binary_") ||
               STARTS_WITH(task.get_task_name(), "readTileAttr") ||
               STARTS_WITH(task.get_task_name(), "AffineTransform")) {
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
      // Work tasks: map to IO processors, so they don't get blocked by tiny
      // CPU tasks.
      if (EQUALS(task.get_task_name(), "workSingle") ||
          EQUALS(task.get_task_name(), "workDual")) {
         ranking.push_back(Processor::IO_PROC);
      }
      // Other tasks: defer to the default mapping policy
      else {
         DefaultMapper::default_policy_rank_processor_kinds(ctx, task, ranking);
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
      SplinteringFunctor* functor = pick_functor(ctx, task);
      for (Domain::DomainPointIterator it(input.domain); it; it++) {
         Processor target_proc = select_proc(it.p, info.proc_kind, functor);
         output.slices.emplace_back(Domain(it.p, it.p), target_proc,
                                    false/*recurse*/, false/*stealable*/);
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
      // Increase priority of tasks on the critical path of the fluid solve.
      if (STARTS_WITH(task.get_task_name(), "Exports.GetVelocityGradients") ||
          STARTS_WITH(task.get_task_name(), "Exports.GetEulerFlux") ||
          STARTS_WITH(task.get_task_name(), "GetFlux") ||
          STARTS_WITH(task.get_task_name(), "UpdateUsingFlux") ||
          STARTS_WITH(task.get_task_name(), "CorrectUsingFlux") ||
          STARTS_WITH(task.get_task_name(), "UpdateVars") ||
          STARTS_WITH(task.get_task_name(), "Exports.UpdateChemistry")) {
         priority = 1;
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

   // Send each fill to the rank of its parent task.
   virtual void select_sharding_functor(const MapperContext ctx,
                                        const Fill& fill,
                                        const SelectShardingFunctorInput& input,
                                        SelectShardingFunctorOutput& output) {
      output.chosen_functor = pick_functor(ctx, *(fill.parent_task))->id;
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
                                                      const RegionRequirement& req) {
      return DefaultMapper::default_policy_select_target_memory(ctx, target_proc, req);
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

     if (EQUALS(name, "p_Interior") ||
         EQUALS(name, "p_Fluid_AllGhost") ||
         EQUALS(name, "p_x_divg") ||
         EQUALS(name, "p_y_divg") ||
         EQUALS(name, "p_z_divg") ||
         EQUALS(name, "p_solved") ||
         EQUALS(name, "p_x_faces") ||
         EQUALS(name, "p_y_faces") ||
         EQUALS(name, "p_z_faces")) {
        DomainPoint tile = runtime->get_logical_region_color_point(ctx, req.region);
        LOG.debug() << "Region " << name
                    << "[Tile " << tile << "] "
                    << "is mapped on corresponding instance of p_All";
        LogicalRegion root_region = get_root(ctx, req.region);
        LogicalPartition primary_partition = get_primary_partition(ctx, root_region);
        assert(primary_partition != LogicalPartition::NO_PART);
        return runtime->get_logical_subregion_by_color(ctx, primary_partition, tile);
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
   }

   // Disable an optimization done by the default mapper (extends the set of
   // eligible processors to include all the processors of the same type on the
   // target node).
   virtual void default_policy_select_target_processors(MapperContext ctx,
                                                        const Task &task,
                                                        std::vector<Processor> &target_procs) {
      target_procs.push_back(task.target_proc);
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
   //virtual void select_sharding_functor(const MapperContext ctx,
   //                                     const Partition& partition,
   //                                     const SelectShardingFunctorInput& input,
   //                                     SelectShardingFunctorOutput& output) {
   //  CHECK(false, "Unsupported: Sharded Partition");
   //}
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
  // NOTE: This function doesn't sanity check its input.
  Processor select_proc(const DomainPoint& tile,
                        Processor::Kind kind,
                        SplinteringFunctor* functor) {
      AddressSpace rank = functor->get_rank(tile);
      const std::vector<Processor>& procs = get_procs(rank, kind);
      SplinterID splinter_id = functor->splinter(tile);
      return procs[splinter_id % procs.size()];
   }

   std::vector<Processor>& get_procs(AddressSpace rank, Processor::Kind kind) {
      assert(rank < all_procs_.size());
      auto& rank_procs = all_procs_[rank];
      if (kind >= rank_procs.size()) {
         rank_procs.resize(kind + 1);
      }
      return rank_procs[kind];
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

   LogicalPartition get_primary_partition(MapperContext ctx,
                                          const LogicalRegion &region) {
     std::set<Color> colors;
     runtime->get_index_space_partition_colors(ctx, region.get_index_space(), colors);
     for (std::set<Color>::const_iterator it = colors.begin(); it != colors.end(); ++it) {
       LogicalPartition lp = runtime->get_logical_partition_by_color(ctx, region, *it);
       const char *name = get_partition_name(ctx, lp);
       if (name != NULL && EQUALS(name, "p_All")) {
         return lp;
       }
     }
     return LogicalPartition::NO_PART;
   }

//=============================================================================
// MAPPER CLASS: MEMBER VARIABLES
//=============================================================================

private:
   std::deque<SampleMapping> sample_mappings_;
   std::vector<std::vector<std::vector<Processor> > > all_procs_;
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
