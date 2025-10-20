---
title: "Single Writer Multiple Reader (SWMR) — h5py 3.13.0 documentation"
source: "https://docs.h5py.org/en/stable/swmr.html"
author:
published:
created: 2025-02-23
description:
tags:
  - "clippings"
---
Starting with version 2.5.0, h5py includes support for the HDF5 SWMR features.

## What is SWMR?

The SWMR features allow simple concurrent reading of a HDF5 file while it is being written from another process. Prior to this feature addition it was not possible to do this as the file data and meta-data would not be synchronised and attempts to read a file which was open for writing would fail or result in garbage data.

A file which is being written to in SWMR mode is guaranteed to always be in a valid (non-corrupt) state for reading. This has the added benefit of leaving a file in a valid state even if the writing application crashes before closing the file properly.

This feature has been implemented to work with independent writer and reader processes. No synchronisation is required between processes and it is up to the user to implement either a file polling mechanism, inotify or any other IPC mechanism to notify when data has been written.

The SWMR functionality requires use of the latest HDF5 file format: v110. In practice this implies using at least HDF5 1.10 (this can be checked via `h5py.version.info`) and setting the libver bounding to “latest” when opening or creating the file.

Warning

New v110 format files are *not* compatible with v18 format. So, files written in SWMR mode with libver=’latest’ cannot be opened with older versions of the HDF5 library (basically any version older than the SWMR feature).

The HDF Group has documented the SWMR features in details on the website: [Single-Writer/Multiple-Reader (SWMR) Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/_s_w_m_r.html). This is highly recommended reading for anyone intending to use the SWMR feature even through h5py. For production systems in particular pay attention to the file system requirements regarding POSIX I/O semantics.

## Using the SWMR feature from h5py

The following basic steps are typically required by writer and reader processes:

- Writer process creates the target file and all groups, datasets and attributes.
- Writer process switches file into SWMR mode.
- Reader process can open the file with swmr=True.
- Writer writes and/or appends data to existing datasets (new groups and datasets *cannot* be created when in SWMR mode).
- Writer regularly flushes the target dataset to make it visible to reader processes.
- Reader refreshes target dataset before reading new meta-data and/or main data.
- Writer eventually completes and close the file as normal.
- Reader can finish and close file as normal whenever it is convenient.

The following snippet demonstrate a SWMR writer appending to a single dataset:

```default
f = h5py.File("swmr.h5", 'w', libver='latest')
arr = np.array([1,2,3,4])
dset = f.create_dataset("data", chunks=(2,), maxshape=(None,), data=arr)
f.swmr_mode = True
# Now it is safe for the reader to open the swmr.h5 file
for i in range(5):
    new_shape = ((i+1) * len(arr), )
    dset.resize( new_shape )
    dset[i*len(arr):] = arr
    dset.flush()
    # Notify the reader process that new data has been written
```

The following snippet demonstrate how to monitor a dataset as a SWMR reader:

```default
f = h5py.File("swmr.h5", 'r', libver='latest', swmr=True)
dset = f["data"]
while True:
    dset.id.refresh()
    shape = dset.shape
    print( shape )
```

## Examples

In addition to the above example snippets, a few more complete examples can be found in the examples folder. These examples are described in the following sections.

### Dataset monitor with inotify

The inotify example demonstrates how to use SWMR in a reading application which monitors live progress as a dataset is being written by another process. This example uses the the linux inotify ([pyinotify](https://pypi.python.org/pypi/pyinotify) python bindings) to receive a signal each time the target file has been updated.

```default
"""
    Demonstrate the use of h5py in SWMR mode to monitor the growth of a dataset
    on notification of file modifications.

    This demo uses pyinotify as a wrapper of Linux inotify.
    https://pypi.python.org/pypi/pyinotify

    Usage:
            swmr_inotify_example.py [FILENAME [DATASETNAME]]

              FILENAME:    name of file to monitor. Default: swmr.h5
              DATASETNAME: name of dataset to monitor in DATAFILE. Default: data

    This script will open the file in SWMR mode and monitor the shape of the
    dataset on every write event (from inotify). If another application is
    concurrently writing data to the file, the writer must have have switched
    the file into SWMR mode before this script can open the file.
"""
import asyncore
import pyinotify
import sys
import h5py
import logging

#assert h5py.version.hdf5_version_tuple >= (1,9,178), "SWMR requires HDF5 version >= 1.9.178"

class EventHandler(pyinotify.ProcessEvent):

    def monitor_dataset(self, filename, datasetname):
        logging.info("Opening file %s", filename)
        self.f = h5py.File(filename, 'r', libver='latest', swmr=True)
        logging.debug("Looking up dataset %s"%datasetname)
        self.dset = self.f[datasetname]

        self.get_dset_shape()

    def get_dset_shape(self):
        logging.debug("Refreshing dataset")
        self.dset.refresh()

        logging.debug("Getting shape")
        shape = self.dset.shape
        logging.info("Read data shape: %s"%str(shape))
        return shape

    def read_dataset(self, latest):
        logging.info("Reading out dataset [%d]"%latest)
        self.dset[latest:]

    def process_IN_MODIFY(self, event):
        logging.debug("File modified!")
        shape = self.get_dset_shape()
        self.read_dataset(shape[0])

    def process_IN_CLOSE_WRITE(self, event):
        logging.info("File writer closed file")
        self.get_dset_shape()
        logging.debug("Good bye!")
        sys.exit(0)

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s  %(levelname)s\t%(message)s',level=logging.INFO)

    file_name = "swmr.h5"
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
    dataset_name = "data"
    if len(sys.argv) > 2:
        dataset_name = sys.argv[2]

    wm = pyinotify.WatchManager()  # Watch Manager
    mask = pyinotify.IN_MODIFY | pyinotify.IN_CLOSE_WRITE
    evh = EventHandler()
    evh.monitor_dataset( file_name, dataset_name )

    notifier = pyinotify.AsyncNotifier(wm, evh)
    wdd = wm.add_watch(file_name, mask, rec=False)

    # Sit in this loop() until the file writer closes the file
    # or the user hits ctrl-c
    asyncore.loop()
```

### Multiprocess concurrent write and read

The SWMR multiprocess example starts two concurrent child processes: a writer and a reader. The writer process first creates the target file and dataset. Then it switches the file into SWMR mode and the reader process is notified (with a multiprocessing.Event) that it is safe to open the file for reading.

The writer process then continue to append chunks to the dataset. After each write it notifies the reader that new data has been written. Whether the new data is visible in the file at this point is subject to OS and file system latencies.

The reader first waits for the initial “SWMR mode” notification from the writer, upon which it goes into a loop where it waits for further notifications from the writer. The reader may drop some notifications, but for each one received it will refresh the dataset and read the dimensions. After a time-out it will drop out of the loop and exit.

```default
"""
    Demonstrate the use of h5py in SWMR mode to write to a dataset (appending)
    from one process while monitoring the growing dataset from another process.

    Usage:
            swmr_multiprocess.py [FILENAME [DATASETNAME]]

              FILENAME:    name of file to monitor. Default: swmrmp.h5
              DATASETNAME: name of dataset to monitor in DATAFILE. Default: data

    This script will start up two processes: a writer and a reader. The writer
    will open/create the file (FILENAME) in SWMR mode, create a dataset and start
    appending data to it. After each append the dataset is flushed and an event
    sent to the reader process. Meanwhile the reader process will wait for events
    from the writer and when triggered it will refresh the dataset and read the
    current shape of it.
"""

import sys
import h5py
import numpy as np
import logging
from multiprocessing import Process, Event

class SwmrReader(Process):
    def __init__(self, event, fname, dsetname, timeout = 2.0):
        super().__init__()
        self._event = event
        self._fname = fname
        self._dsetname = dsetname
        self._timeout = timeout

    def run(self):
        self.log = logging.getLogger('reader')
        self.log.info("Waiting for initial event")
        assert self._event.wait( self._timeout )
        self._event.clear()

        self.log.info("Opening file %s", self._fname)
        f = h5py.File(self._fname, 'r', libver='latest', swmr=True)
        assert f.swmr_mode
        dset = f[self._dsetname]
        try:
            # monitor and read loop
            while self._event.wait( self._timeout ):
                self._event.clear()
                self.log.debug("Refreshing dataset")
                dset.refresh()

                shape = dset.shape
                self.log.info("Read dset shape: %s"%str(shape))
        finally:
            f.close()

class SwmrWriter(Process):
    def __init__(self, event, fname, dsetname):
        super().__init__()
        self._event = event
        self._fname = fname
        self._dsetname = dsetname

    def run(self):
        self.log = logging.getLogger('writer')
        self.log.info("Creating file %s", self._fname)
        f = h5py.File(self._fname, 'w', libver='latest')
        try:
            arr = np.array([1,2,3,4])
            dset = f.create_dataset(self._dsetname, chunks=(2,), maxshape=(None,), data=arr)
            assert not f.swmr_mode

            self.log.info("SWMR mode")
            f.swmr_mode = True
            assert f.swmr_mode
            self.log.debug("Sending initial event")
            self._event.set()

            # Write loop
            for i in range(5):
                new_shape = ((i+1) * len(arr), )
                self.log.info("Resizing dset shape: %s"%str(new_shape))
                dset.resize( new_shape )
                self.log.debug("Writing data")
                dset[i*len(arr):] = arr
                #dset.write_direct( arr, np.s_[:], np.s_[i*len(arr):] )
                self.log.debug("Flushing data")
                dset.flush()
                self.log.info("Sending event")
                self._event.set()
        finally:
            f.close()

if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)10s  %(asctime)s  %(name)10s  %(message)s',level=logging.INFO)
    fname = 'swmrmp.h5'
    dsetname = 'data'
    if len(sys.argv) > 1:
        fname = sys.argv[1]
    if len(sys.argv) > 2:
        dsetname = sys.argv[2]

    event = Event()
    reader = SwmrReader(event, fname, dsetname)
    writer = SwmrWriter(event, fname, dsetname)

    logging.info("Starting reader")
    reader.start()
    logging.info("Starting reader")
    writer.start()

    logging.info("Waiting for writer to finish")
    writer.join()
    logging.info("Waiting for reader to finish")
    reader.join()
```

The example output below (from a virtual Ubuntu machine) illustrate some latency between the writer and reader:

```default
python examples/swmr_multiprocess.py
  INFO  2015-02-26 18:05:03,195        root  Starting reader
  INFO  2015-02-26 18:05:03,196        root  Starting reader
  INFO  2015-02-26 18:05:03,197      reader  Waiting for initial event
  INFO  2015-02-26 18:05:03,197        root  Waiting for writer to finish
  INFO  2015-02-26 18:05:03,198      writer  Creating file swmrmp.h5
  INFO  2015-02-26 18:05:03,203      writer  SWMR mode
  INFO  2015-02-26 18:05:03,205      reader  Opening file swmrmp.h5
  INFO  2015-02-26 18:05:03,210      writer  Resizing dset shape: (4,)
  INFO  2015-02-26 18:05:03,212      writer  Sending event
  INFO  2015-02-26 18:05:03,213      reader  Read dset shape: (4,)
  INFO  2015-02-26 18:05:03,214      writer  Resizing dset shape: (8,)
  INFO  2015-02-26 18:05:03,214      writer  Sending event
  INFO  2015-02-26 18:05:03,215      writer  Resizing dset shape: (12,)
  INFO  2015-02-26 18:05:03,215      writer  Sending event
  INFO  2015-02-26 18:05:03,215      writer  Resizing dset shape: (16,)
  INFO  2015-02-26 18:05:03,215      reader  Read dset shape: (12,)
  INFO  2015-02-26 18:05:03,216      writer  Sending event
  INFO  2015-02-26 18:05:03,216      writer  Resizing dset shape: (20,)
  INFO  2015-02-26 18:05:03,216      reader  Read dset shape: (16,)
  INFO  2015-02-26 18:05:03,217      writer  Sending event
  INFO  2015-02-26 18:05:03,217      reader  Read dset shape: (20,)
  INFO  2015-02-26 18:05:03,218      reader  Read dset shape: (20,)
  INFO  2015-02-26 18:05:03,219        root  Waiting for reader to finish
```