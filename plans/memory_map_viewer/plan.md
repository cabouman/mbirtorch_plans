## Memory-map large volumes in the viewers

In the mbirtorch slice_viewer, viewing large arrays 
loads them fully into memory.    

This is also relevant for pcdrecon on the GUI thread: the Datasets tab's seam function
pcd.gui.datasets.load_counts (its docstring says memory-mapping is a change to that 
function alone; a 2.9 GB scan currently takes ~5 s and blocks the window), and the Run 
history tab's View arrays path loads saved .npy files the same way. 

Greg wants memory-mapped viewing investigated for large volumes generally. 
First determine whether the slice viewer (pcd/view.py over the mbirtorch slice viewer) 
indexes arrays lazily or materializes them — that decides everything. If lazy: wire 
np.memmap into load_counts (a raw .scan needs shape from the scan's CatSim_config.mat 
and the bins-fastest stored order, then a lazily-transposed view; verify the scan 
loader's reshape semantics in pcd/data_read_write/scan_io.py load_pcd_data before 
reimplementing them) and np.load(..., mmap_mode='r') into the Run history array views; 
keep the size-confirmation dialog only for the code paths that still materialize (e.g. 
the transmission divide, which computes a new array — consider computing it per-slice in 
the viewer instead, or keep the confirmation there). If the viewer materializes: report 
what it would take to make it lazy and stop for Greg's direction. Environment python: 
/Users/gbuzzard/miniforge3/envs/pcdrecon/bin/python; suite at 187 passing; stage nothing 
(Greg commits himself).