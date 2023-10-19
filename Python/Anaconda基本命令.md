## 1、创建虚拟环境

```anaconda
conda create -n xxx python=3.8
```

## 2、导出虚拟环境

```anaconda
conda env export --name xxx > myenv.yml
```

## 3、导入虚拟环境

```anaconda
conda env create -f myenv.yml
```

## 4、为虚拟环境添加 Jupyter Lab

```
conda activate myenv
conda install jupyter
python -m ipykernel install --user --name=myenv
```

## 5、虚拟环境列表

```bash
conda env list
```

