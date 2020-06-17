import torch, fastai, sys, os
from fastai.vision import *
from fastai.vision.data import SegmentationProcessor
import ants
from ants.core.ants_image import ANTsImage
from jupyterthemes import jtplot
sys.path.insert(0, './exp')
jtplot.style(theme='gruvboxd')

# Set a root directory
path = Path('/home/ubuntu/MultiCampus/MICCAI_BraTS_2019_Data_Training')

def is_mod(fn:str, mod:str)->bool:
    "Check if file path contains a specified name of modality used for MRI"
    import re
    r = re.compile('.*' + mod, re.IGNORECASE)
    return True if r.match(fn) else False

def is_mods(fn:str, mods:Collection[str])->bool:
    "Check if file path contains specified names of modality used for MRI"
    import re
    return any([is_mod(fn, mod) for mod in mods])

def _path_to_same_str(p_fn):
    "path -> str, but same on nt+posix, for alpha-sort only"
    s_fn = str(p_fn)
    s_fn = s_fn.replace('\\','.')
    s_fn = s_fn.replace('/','.')
    return s_fn

def _get_files(path, file, modality):
    """
    Internal implementation for `get_files` to combine a parent directory with a file 
    to make a full path to file(s)
    """
    p = Path(path)
    res = [p/o for o in file if not o.startswith('.') and is_mods(o, modality)]
    assert len(res)==len(modality) #TODO: Assert message
    return res

def get_files(path:PathOrStr, modality:Union[str, Collection[str]], 
                presort:bool=False)->FilePathList:
    "Return a list of full file paths in `path` each of which contains modality in its name"
    file = [o.name for o in os.scandir(path) if o.is_file()]
    res = _get_files(path, file, modality)
    if presort: res = sorted(res, key=lambda p: _path_to_same_str(p), reverse=False)
    return res

def _repr_antsimage(self):
    if self.dimension == 3:
        s = 'NiftiImage ({})\n'.format(self.orientation)
    else:
        s = 'NiftiImage\n'
    s = s +\
        '\t {:<10} : {} ({})\n'.format('Pixel Type', self.pixeltype, self.dtype)+\
        '\t {:<10} : {}{}\n'.format('Components', self.components, ' (RGB)' if 'RGB' in self._libsuffix else '')+\
        '\t {:<10} : {}\n'.format('Dimensions', self.shape)+\
        '\t {:<10} : {}\n'.format('Spacing', tuple([round(s,4) for s in self.spacing]))+\
        '\t {:<10} : {}\n'.format('Origin', tuple([round(o,4) for o in self.origin]))+\
        '\t {:<10} : {}\n'.format('Direction', np.round(self.direction.flatten(),4))
    return s

# Modify the representation of `ANTsImage` object
ANTsImage.__repr__ = _repr_antsimage

class NiftiImage(ItemBase):
  "Support handling NIfTI image format" 
  #TODO: Extend the code so as to support various Python (medical) libraries that can read NIfTI format   
  def __init__(self, data:Union[Tensor,np.array], obj:ANTsImage, path:str): 
    self.data = data
    self.obj = obj
    self.path = path
    # Only works for a specific folder tree
    self.mod = self.path.split(".")[0].split("_")[-1]
  
  def __repr__(self): return str(self.obj) + '\t {:<10} : {}\n\n'.format('Modality', str(self.mod))

  def __getattr__(self, k:str):
    func = getattr(self.obj, k)
    if isinstance(func, Callable): return func
  
  def __setattr__(self, k, v):
    if k == 'obj':
        self.data = torch.tensor(v.numpy())
    return super().__setattr__(k, v)

  # This wraps ANTsPy's `plot` method to show NIfTI image
  def show(self, **kwargs):
    ants.plot(self.obj)

  # This wraps ANTsPy's `image_read` method to read NIfTI format
  @classmethod
  def create(cls, path:PathOrStr):
    nimg = ants.image_read(str(path))
    t = torch.tensor(nimg.numpy())
    return cls(t, nimg, path)

  def apply_tfms(self, tfms:List[Transform], *args, order='order', **kwargs):
    key = lambda o : getattr(o, order, 0)
    for tfm in sorted(listify(tfms), key=key): self = tfm(self, *args, **kwargs) #ascending order eg. [3,2,1] -> [1,2,3]
    return self

class MultiNiftiImage(ItemBase):
  "Support handling multi-channel NIfTI images"
  def __init__(self, obj:Tuple[NiftiImage]):
    self.obj = obj # type annotation violated when `subregionify` is used. Should be fixed.
    self.data = None
  
  def __repr__(self): 
        return f"Inside {self.__class__.__name__}:\n {[self.obj[i] for i in range(len(self.obj))]}"       
   
  def __getitem__(self, i):
        return self.obj[i]
        
  @classmethod
  def create(cls, paths:FilePathList):
    obj = tuple([NiftiImage.create(str(path)) for path in paths])
    return cls(obj)

  def apply_tfms(self, tfms:List[Transform], *args, order='order', **kwargs):
    self.obj = tuple([self.obj[i].apply_tfms(tfms, order, *args, **kwargs) for i in range(len(self.obj))])
    self.data = torch.stack([nft.data for nft in self.obj], dim=0)
    return self

  @property
  def data(self):
    return self._data

  @data.setter
  def data(self, _):
    self._data = ( torch.stack([nft.data for nft in self.obj], dim=0) 
                  if hasattr(self.obj[0], "data") 
                  else torch.stack([torch.tensor(nft.numpy()) for nft in self.obj], dim=0) )

class NiftiImageList(ItemList):
     
  def __repr__(self)->str: 
    return '{} ({} items)\n{}\nPath: {}'.format(self.__class__.__name__, 
                                                len(self.items), show_some(self.items, n_max=4, sep="\n"), 
                                                self.path)  
  def get(self, i)->NiftiImage:
    fn = str(self.items[i])
    return NiftiImage.create(fn)

class MultiNiftiImageList(ItemList):

  def __repr__(self)->str: 
    return '{} ({} items)\n{}\nPath: {}'.format(self.__class__.__name__, 
                                                len(self.items), show_some(self.items, n_max=4, sep="\n"), 
                                                self.path)  
  def get(self, i)->MultiNiftiImage:
    filepaths = [str(self.items[i][x]) for x in range(len(self.items[i]))]
    return MultiNiftiImage.create(filepaths)

  @classmethod
  def from_folder(cls, folderpaths:FilePathList, modality:Union[str, Collection[str]], 
                  presort:bool=False, **kwargs):
    """
    This method assumes a list of full paths to the desired files's parent folders 
    and returns NiftiImageTupleList whose item is a nested list with each sublist 
    belonging to its parent folder
    -------------------------------------------------------------------------
        Test:
        assert len(filepaths) == len(path)
        
    """
    filepaths=[]
    for fp in folderpaths:
      filepath = get_files(fp, modality=modality, presort=True)
      filepaths.append(filepath)
        
    return cls(items=filepaths, path=path, **kwargs)

hgg_subdirs = (path/'HGG').ls()
lgg_subdirs = (path/'LGG').ls()
parent_folders = hgg_subdirs + lgg_subdirs

def get_parents(path:Path, pname:str, shuffle:bool=True, pct=0.2):
  "List a certain percent of items under a specified parent directory randomly or not"
  from random import shuffle
  ps = [d[i] for r,d,_ in os.walk(path) for i in range(len(d)) if Path(r).name==pname] 
  if shuffle: shuffle(ps)
  return ps[:round((pct*len(ps)))]

def write_val_list(fname:str='valid.txt', vals:List[str]=None):
  "Write a list of names into `fname` to be used for train/validation split"
  val_list = vals
  with open(fname, 'w') as f:
    f.write('\n'.join(val_list))
  print("{} items written into {}.".format(len(val_list), fname))

val_list = get_parents(path, 'HGG', pct=0.15) + get_parents(path, 'LGG', pct=0.1)
write_val_list('valid.txt', val_list)

def split_by_parents(self, valid_names:'ItemList')->'ItemLists':
  "Split the data by using the parent names in `valid_names` for validation."
  return self.split_by_valid_func(lambda o: o.parent.name in valid_names)

def split_by_pname_file(self, fname:PathOrStr, path:PathOrStr=None)->'ItemLists':
  "Split the data by using the parent names in `fname` for the validation set. `path` will override `self.path`."
  path = Path(ifnone(path, self.path))
  valid_names = loadtxt_str(path/fname)
  return self.split_by_parents(valid_names) 

def split_by_valid_func(self, func:Callable)->'ItemLists':
  "Split the data by result of `func` (which returns `True` for validation set)."
  valid_idx = [i for i,o in enumerate(self.items) if func(o[0])]
  return self.split_by_idx(valid_idx)
    
def _repr_labellist(self)->str:
  items = [self[i] for i in range(min(1,len(self.items)))]
  res = f'{self.__class__.__name__} ({len(self.items)} items)\n'
  res += f'x: {self.x.__class__.__name__}\n{show_some([i[0] for i in items], n_max=1)}\n'
  res += f'y: {self.y.__class__.__name__}\n{show_some([i[1] for i in items], n_max=1)}\n'
  return res + f'Path: {self.path}'

# Modify the methods of `MultiNiftiImageList` object
MultiNiftiImageList.split_by_parents = split_by_parents
MultiNiftiImageList.split_by_pname_file = split_by_pname_file
MultiNiftiImageList.split_by_valid_func = split_by_valid_func

# Modify the representation of `LabelList` object
LabelList.__repr__ = _repr_labellist

class NiftiSegmentationLabelList(NiftiImageList):
  "`ItemList` for NIfTI segmentatoin masks"
  _processor=SegmentationProcessor
    
  def __init__(self, items:Iterator, classes:Collection=None, **kwargs):
    super().__init__(items, **kwargs)
    self.copy_new.append('classes')
    self.classes,self.loss_func = classes,None
    
  def reconstruct(self, t:Tensor): 
    obj = ants.from_numpy(t.numpy())
    path = self.path
    return NiftiImage(t, obj, path)

get_y_fn = lambda x: x[0].parent/Path(x[0].as_posix().split(os.sep)[-2]+'_seg.nii.gz')

subregion = np.array(['WT', 'TC', 'ET']) 

def crop_3d(item:NiftiImage, do_resolve=False, *args, lowerind:Tuple, upperind:Tuple, **kwargs):
  "Crop 3-dimensional NIfTI image by slicing indices from lower to upper indices per image axis"
  cropped_item = item.obj.crop_indices(lowerind, upperind)
  item.obj = cropped_item
  return item

def standardize(item:NiftiImage, do_resolve=False, *args, **kwargs):
  "Standardize our custom itembase `NiftiImage` to have zero mean and unit std based on non-zero voxels only"
  arr = item.obj.numpy()
  arr_nonzero = arr[arr!=0]
  arr_nonzero = (arr_nonzero - arr_nonzero.mean()) / arr_nonzero.std()
  arr[arr!=0] = arr_nonzero / arr_nonzero.max()
  item.obj = ants.from_numpy(arr)
  return item

def subregionify(item:NiftiImage, do_resolve=False, *args, **kwargs): 
  "Combine the three annotations into 3 nested subregions: Whole Tumor(WT), Tumor Core(TC), Enhancing Tumor(ET)"
  arr = item.obj.numpy()
  wt_arr = arr.copy()
  wt_arr[wt_arr==1.] = 1.; wt_arr[wt_arr==2.] = 1.; wt_arr[wt_arr==4.] = 1.
  tc_arr = arr.copy()
  tc_arr[tc_arr==1.] = 1.; tc_arr[tc_arr==2.] = 0.; tc_arr[tc_arr==4.] = 1.
  et_arr = arr.copy()
  et_arr[et_arr==1.] = 0.; et_arr[et_arr==2.] = 0.; et_arr[et_arr==4.] = 1.
  return MultiNiftiImage([ants.from_numpy(arr) for arr in [wt_arr, tc_arr, et_arr]])

crop_3d = Transform(crop_3d, order=0)            # Applied to 'x' first then `y` for a implementation detail with overwrite
standardize = Transform(standardize, order=1)    # Only applied to 'x'
subregionify = Transform(subregionify, order=1)  # Only applied to 'y'

x_transform = [crop_3d, standardize]
y_transform = [crop_3d, subregionify]

data = (MultiNiftiImageList.from_folder(parent_folders, modality=['Flair', 'T1', 'T2', 'T1ce'])
               .split_by_pname_file(fname='valid.txt', path=Path('.'))
               .label_from_func(get_y_fn, classes=subregion, label_cls=NiftiSegmentationLabelList)
               .transform((x_transform, x_transform), tfm_y=False, lowerind=(40,28,10), upperind=(200,220,138))
               .transform_y((y_transform, y_transform), lowerind=(40,28,10), upperind=(200,220,138))
               .databunch(bs=1, collate_fn=data_collate, num_workers=0))