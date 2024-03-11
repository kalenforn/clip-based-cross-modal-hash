from torch.utils.data import Dataset

# @registry.register_dataset("base_dataset")

class BaseDataset(Dataset):

    def _load_image(self, index: int):
        raise NotImplementedError()
    
    def __len__(self):
        raise NotImplementedError()
    
    def _load_text(self, index: int):
        raise NotImplementedError()
    
    def _load_label(self, index: int):
        raise NotImplementedError()
    
    def get_all_label(self):
        raise NotImplementedError()

    def __getitem__(self, index):
        raise NotImplementedError()
    
    def get_tag_length(self):
        return self.captions.shape[-1]

