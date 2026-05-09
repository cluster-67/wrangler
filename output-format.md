```bash
> h5ls -rv /pscratch/sd/a/ac3354/data/fashion-mnist-784-euclidean.hdf5
Opened "/pscratch/sd/a/ac3354/data/fashion-mnist-784-euclidean.hdf5" with sec2 driver.
/                        Group
    Attribute: distance scalar
        Type:      variable-length null-terminated UTF-8 string
    Location:  1:96
    Links:     1
/distances               Dataset {10000/10000, 100/100}
    Location:  1:219526240
    Links:     1
    Storage:   4000000 logical bytes, 4000000 allocated bytes, 100.00% utilization
    Type:      native float
/neighbors               Dataset {10000/10000, 100/100}
    Location:  1:1800
    Links:     1
    Storage:   4000000 logical bytes, 4000000 allocated bytes, 100.00% utilization
    Type:      native int
/test                    Dataset {10000/10000, 784/784}
    Location:  1:1528
    Links:     1
    Storage:   31360000 logical bytes, 31360000 allocated bytes, 100.00% utilization
    Type:      native float
/train                   Dataset {60000/60000, 784/784}
    Location:  1:928
    Links:     1
    Storage:   188160000 logical bytes, 188160000 allocated bytes, 100.00% utilization
    Type:      native float
> h5dump -a /distance /pscratch/sd/a/ac3354/data/fashion-mnist-784-euclidean.hdf5
HDF5 "/pscratch/sd/a/ac3354/data/fashion-mnist-784-euclidean.hdf5" {
ATTRIBUTE "distance" {
   DATATYPE  H5T_STRING {
      STRSIZE H5T_VARIABLE;
      STRPAD H5T_STR_NULLTERM;
      CSET H5T_CSET_UTF8;
      CTYPE H5T_C_S1;
   }
   DATASPACE  SCALAR
   DATA {
   (0): "euclidean"
   }
}
}
```
