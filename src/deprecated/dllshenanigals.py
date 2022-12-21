def is_loaded(lib):
  libp = os.path.abspath(lib)
  cmd = "lsof -p %d | grep %s > /dev/null" % (os.getpid(), libp)
  # ipdb.set_trace()
  # print(cmd)
  ret = os.system(cmd)
  print("ret=" , ret)
  return (ret == 0)

def reload_lib(lib):
  handle = lib._handle
  name = lib._name
  del lib
  while is_loaded(name):   
    libdl = ctypes.CDLL("zig-out/lib/libtrack.dylib")
    # handle = libdl._handle
    libdl.dlclose(handle)
  return ctypes.cdll.LoadLibrary(name)

def unload_lib(lib):
  handle = lib._handle
  name = lib._name
  del lib
  while is_loaded(name):
    print(name)
    # libdl = ctypes.CDLL("zig-out/lib/libtrack.dylib")
    # handle = libdl._handle
    libdl.dlclose(handle)

def libload():
  return ctypes.cdll.LoadLibrary("zig-out/lib/libtrack.dylib")

def libtest():
  # get the module handle and create a ctypes library object
  # libHandle = ctypes.windll.kernel32.LoadLibraryA('mydll.dll')
  lib = ctypes.CDLL("zig/zig-out/lib/libtrack.dylib")
  print(lib.strain_track2d)
  print(lib.greedy_track2d)

  # clean up by removing reference to the ctypes library object
  # del lib

  # unload the DLL
  # ctypes.windll.kernel32.FreeLibrary(libHandle)
  # ctypes.cdll.FreeLibrary

