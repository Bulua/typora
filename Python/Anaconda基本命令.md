[toc]


#### 0、镜像源
```anaconda
清华：https://pypi.tuna.tsinghua.edu.cn/simple
阿里云：http://mirrors.aliyun.com/pypi/simple/
中国科技大学 https://pypi.mirrors.ustc.edu.cn/simple/
华中理工大学：http://pypi.hustunique.com/
山东理工大学：http://pypi.sdutlinux.org/
豆瓣：http://pypi.douban.com/simple/
```

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
conda info --envs
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

#### 12、更新pip

```bash
python -m pip install --upgrade pip
```

#### 13、配置下载通道

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 14、清除缓存

1. **清理软件包缓存：**

   ```bash
   conda clean -p
   ```

   这个命令将清理软件包缓存。

2. **清理索引缓存：**

   ```bash
   conda clean -i
   ```

   这个命令将清理索引缓存。

3. **清理锁文件：**

   ```bash
   conda clean -l
   ```

   这个命令将清理`conda`的锁定文件。

4. **清理所有缓存：**

   ```bash
   conda clean --all
   ```

   这个命令将清理所有软件包、索引和锁文件缓存。

