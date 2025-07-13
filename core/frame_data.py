from easydict import EasyDict
import numpy as np

class FrameData():
    def __init__(self, points=None):
        super().__init__()
        self.pcd = EasyDict({
            'points': points,
            'colors': None,
        })
        self.geometry = EasyDict({
            'arrows': [],
            'spheres': [],
            'boxes': [],
        })
        self.pose = EasyDict({
            'R': None,
            'T': None,
        })
    
    def reset(self):
        """
        重置数据
        """
        self.pcd.points = None
        self.pcd.colors = None
        self.geometry.arrows.clear()
        self.geometry.spheres.clear()
        self.geometry.boxes.clear()
        self.pose.R = None
        self.pose.T = None
        # 删除所有其他属性
        for key in list(self.__dict__.keys()):
            if key not in ['pcd', 'geometry', 'pose']:
                del self.__dict__[key]
    
    def __setattr__(self, key, value):
        """
        支持通过 frame_data.属性名 = value 设置属性
        """
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            # 如果属性不存在，则添加新的属性
            self.__dict__[key] = value
    
    
    def __get_basic_type_str(self, name: str, value, indent: int = 0) -> str:
        result = ' ' * indent + f"{name}: {type(value)}"
        if isinstance(value, np.ndarray):
            result += f" shape={value.shape},\n"
        elif isinstance(value, list):
            result += f" len={len(value)},\n"
        elif isinstance(value, str):
            result += f" value='{value}',\n"
        elif isinstance(value, int):
            result += f" value={value},\n"
        elif isinstance(value, float):
            result += f" value={value},\n"
        elif isinstance(value, bool):
            result += f" value={value},\n"
        else:
            result += ",\n"
        return result

    
    def __get_dict_str(self, name: str, d: EasyDict, indent: int = 0) -> str:
        result = ' ' * indent + f"{name}: " + "{\n"
        for key, value in d.items():
            if isinstance(value, EasyDict) or isinstance(value, dict):
                result += self.__get_dict_str(f"{key}", value, indent + 2)
            else:
                result += self.__get_basic_type_str(f"{key}", value, indent + 2)
        result += ' ' * indent + "}\n"
        return result


    def __repr__(self):
        result = "FrameData{\n"
        result += self.__get_dict_str("pcd", self.pcd, indent=2)
        result += self.__get_dict_str("geometry", self.geometry, indent=2)
        result += self.__get_dict_str("pose", self.pose, indent=2)
        for key in self.__dict__:
            if key not in ['pcd', 'geometry', 'pose']:
                if isinstance(self.__dict__[key], EasyDict) or isinstance(self.__dict__[key], dict):
                    result += self.__get_dict_str(f"{key}", self.__dict__[key], indent=2)
                else:
                    result += self.__get_basic_type_str(f"{key}", self.__dict__[key], indent=2)
        result += "}"
        return result
