import ctypes

import numpy as np
import ipdb 
from numpy.ctypeslib import ndpointer
import os
import sys
import ctypes

from matplotlib import pyplot as plt
plt.ion()

import os
from glob import glob


def strain_track(va,vb):
  va = va.astype(np.float32)
  vb = vb.astype(np.float32)
  va_ = va.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
  vb_ = vb.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
  parents = np.zeros(len(vb), dtype=np.int32)
  res_ = parents.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
  # ipdb.set_trace()
  err = lib.strain_track2d(va_ , len(va) , vb_ , len(vb) , res_)
  if (err!=0): print("ERRORRRRRROROR")
  return parents

def greedy_track(va,vb):
  va = va.astype(np.float32)
  vb = vb.astype(np.float32)
  va_ = va.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
  vb_ = vb.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
  parents = np.zeros(len(vb), dtype=np.int32)
  res_ = parents.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
  # err = lib.greedy_track2d.restype = ndpointer(dtype=ctypes.c_int32, shape=(len(vb),))
  err = lib.greedy_track2d(va_,len(va),vb_,len(vb), res_)
  if (err!=0): print("ERRORRRRRROROR")
  return parents


def loadlib():
  search_dir = "/zig-cache/o/"
  files = glob("zig-cache/o/*/libtrack.dylib")
  files.sort(key=lambda x: os.path.getmtime(x))
  return ctypes.CDLL(files[-1])

lib = loadlib()

def test_add():
  print(a)
  print(a.add(0,0))
  print(a.add(0,-1))
  print(a.add(0,2**31-1))
  print(a.add(0,2**31))
  print(a.add(0,2**32))
  print(a.add(0,2**32-1))

def test_sum():
  x = np.arange(1000).astype(np.uint32)
  x_1 = x.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
  print(a.sum(x_1, 1000))

def test_greedy_track2d():
  va = np.random.rand(100,3).astype(dtype=np.float32)
  for alpha in np.arange(10):
    vb = va + np.random.rand(100,3).astype(dtype=np.float32) * 0.003*alpha
    p = greedy_track(va,vb)
    plt.plot(p)

def test_greedy_track2d_2(N):
  va = np.random.rand(100,3).astype(dtype=np.float32)
  # for alpha in range(10):
  # N = 100 + alpha*10
  vb = np.random.rand(N,3).astype(dtype=np.float32)
  p = greedy_track(va,vb)
  plt.plot(p)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def test_strain_track(N,viewer):
  va = 500 * np.random.rand(N,2).astype(dtype=np.float32)
  va = va - va.mean(0)
  A = np.array([[1-0.1,0.1],[-0.1,1-0.1]])
  A = A / np.linalg.norm(A,axis=0) #.abs().sum(0,keepdims=True)
  # ipdb.set_trace()
  vb = va.copy()@A
  viewer.add_points(va)
  viewer.add_points(vb,face_color='green')
  res = strain_track(va,vb)
  return res

  # plt.clf()
  # plt.figure()
  # plt.plot(res)
  # plt.show()


if __name__=="__main__":
  test_greedy_track2d_2(10)
  input("wait...")
  test_strain_track(100)
  input("wait some more...")
