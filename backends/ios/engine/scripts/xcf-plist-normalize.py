#!/usr/bin/env python3
"""xcf-plist-normalize.py <LLBEngine.xcframework/Info.plist>: rewrite in place with AvailableLibraries sorted by
LibraryIdentifier. xcodebuild -create-xcframework lists the slices in no fixed order, so the same libraries give
two different Info.plist files; sorted, the file is a function of its content (and equals the vendored pin's)."""
import plistlib, sys
p = sys.argv[1]
d = plistlib.load(open(p, 'rb'))
d['AvailableLibraries'].sort(key=lambda l: l['LibraryIdentifier'])
plistlib.dump(d, open(p, 'wb'), fmt=plistlib.FMT_XML, sort_keys=True)
