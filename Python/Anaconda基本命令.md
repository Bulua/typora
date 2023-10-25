[toc]



#### 1、创建虚拟环境

```anaconda
conda create -n xxx python=3.8
```

如果暂时不想指定python版本可执行：

```bash
conda create --name new_env_name
```

#### 2、导出虚拟环境

```anaconda
conda env export --name xxx > myenv.yml
```

#### 3、导入虚拟环境

```anaconda
conda env create -f myenv.yml
```

#### 4、为虚拟环境添加 Jupyter Lab

```
conda activate myenv
conda install jupyter
python -m ipykernel install --user --name=myenv
```

#### 5、虚拟环境列表

```bash
conda env list
```

#### 6、conda搜索包的版本

这将返回所有pkg_name包的可用版本列表。版本号将显示在每个包的名称后面。

```python
conda search [pkg_name]
```

#### 7、conda搜索已安装包的版本

```bash
conda list [pkg_name]
```

#### 8、conda复制已存在的虚拟环境

```bash
conda create --name new_env_name --clone old_env_name
```

#### 9、删除环境

```bash
conda env remove --name old_env_name
```

或者

```bash
conda env delete --name old_env_name
```

#### 10、激活环境

```bash
conda activate env_name
```

#### 11、退出当前环境

```bash
conda deactivate env_name
```

