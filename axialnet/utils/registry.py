class Registry:
    """
    简单的注册器，用于动态创建类实例。
    """
    def __init__(self, name):
        self._name = name
        self._module_dict = {}

    def register(self, cls):
        """装饰器：@registry.register"""
        self._module_dict[cls.__name__] = cls
        return cls

    def get(self, name):
        """获取类定义"""
        if name not in self._module_dict:
            raise KeyError(f"'{name}' is not registered in {self._name}")
        return self._module_dict[name]

    def build(self, sub_cfg, **kwargs):
        """
        根据配置构建实例
        Args:
            cfg: 包含 'type' 字段的配置字典或 EasyDict
            kwargs: 传递给构造函数的其他参数
        """
        # 兼容 EasyDict 和 dict
        args = sub_cfg.copy()
        obj_type = args.pop('type')
        
        cls = self.get(obj_type)
        return cls(**args, **kwargs)

# 实例化几个常用的注册器
SOLVERS = Registry("solvers")
DATASETS = Registry("datasets")
DATALOADERS = Registry("dataloaders")
MODELS = Registry("models")
BACKBONES = Registry("backbones")
HEADS = Registry("heads")