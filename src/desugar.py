#!/usr/bin/env python2

from itertools import count
import fileinput

for (line, lineno) in zip(fileinput.input(), count(start=1)):
    line = line[:-1]
    line = line.replace('@ESCAPE', '[(function() local __quotes = terralib.newlist()')
    line = line.replace('@EPACSE', 'return __quotes end)()];')
    line = line.replace('@EMIT', '__quotes:insert(rquote')
    line = line.replace('@TIME', 'end)')
    line = line.replace('@LINE', str(lineno))
    print line
