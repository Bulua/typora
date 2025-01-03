[toc]



# nodejs安装

## 1、安装命令

```bash
sudo apt update

sudo apt install -y curl

curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -

sudo apt install nodejs -y
```

## 2、验证是否安装成功

```bash
node -v

npm -v
```

## 3、安装pnpm

```bash
sudo npm install -g pnpm --registry=https://registry.npmmirror.com
```

## 4、验证pnpm

```bash
pnpm -v
```

# 本地部署Dify的Web项目

## 1. 更改docker-compose.yaml

在`docker`目录下的`docker-compose.yaml`文件中，找到`networks` 选项，将`internal: true`注释掉，如下：

```yaml
networks:
  # create a network between sandbox, api and ssrf_proxy, and can not access outside.
  ssrf_proxy_network:
    driver: bridge
    # internal: true
  milvus:
    driver: bridge
  opensearch-net:
    driver: bridge
    # internal: true
```

## 2. 配置dev.conf文件

在`docker/nginx/conf.d`目录下，复制`default.conf`并重新命名为`dev.conf`，并更改监听端口为`3000`。如下：

```conf
server {
    listen 3000;
    server_name _;

    location /console/api {
      proxy_pass http://api:5001;
      include proxy.conf;
    }

    ... 其它内容省略
}
```

## 3. 配置.env.local文件

在`web/`下，复制`.env.example`并重新命名为`.env.local`，更改`NEXT_PUBLIC_API_PREFIX`和`NEXT_PUBLIC_PUBLIC_API_PREFIX`。如下：

```local
NEXT_PUBLIC_API_PREFIX=http://192.168.13.36/console/api
NEXT_PUBLIC_PUBLIC_API_PREFIX=http://192.168.13.36/api
```

其中`192.168.13.36`为本地服务器地址。

## 4. 启动部署

进入到`web`目录

```bash
cd web
```

安装依赖包

```bash
npm install
```

构建代码

```bash
npm run build
```

启动`web`服务

```bash
npm run dev
# or
npm run start
```

# 在Dify中新增自定义工具

## 1. 创建工具文件

在`api/core/tools/provider/builtin`中创建文件夹，**文件夹命名应与工具名称相同**，例如工具名称是`ip_to_adress`，那么应在该文件夹下创建相同名称的`py`和`yaml`文件，以及新增文件夹`_assets`和`tools`。如下：

```bash
api/
└── core/
    └── tools/
        └── provider/
            └── builtin/
                └── ip_to_address/
                    ├── _assets/
                    │   ├── (放置静态资源或辅助文件的地方)
                    ├── tools/
                    │   ├── example_name.py
                    │   └── example_name.yaml
                    ├── ip_to_address.py
                    └── ip_to_address.yaml
```

- **静态资源目录 (`_assets`)**
  - 存放相关的静态资源或辅助文件。
- **工具子目录 (`tools`)**
  - 存放具体实现的工具模块和配置文件。
  - **工具模块 (`example_name.py`)**
    - 示例工具模块文件。
  - **工具配置文件 (`example_name.yaml`)**
    - 示例工具的配置文件。

## 2. 编写相关代码即可

*代码运行有错误会导致工具加载不成功！！！*

## 3. 导入工具所需的相关依赖包

查看容器id

```bash
sudo docker ps
```

进入到容器内

```bash
sudo docker exec -it 容器id /bin/bash
# or
sudo docker exec -it 容器id /bin/sh
```

使用`poetry`下载依赖包（该项目由`poetry`管理而不是`pip`）

```bash
poetry add 包名
```

## 4. 重启项目

进入到`docker`文件夹中，重启

```bash
sudo docker compose down
sudo docker compose up -d
```

## 5. 查看工具页

*如果能够查看到自定义的工具，说明新增成功，否则就是代码环境没有准备好或者文件配置有问题。*

# IPToAdress工具

## 1、纯真IP离线IP数据库

[访问地址](https://cz88.net/geo-public)，官方每周三更新一次，并且可以使用官方给出的下载链接来下载最新的`IP`数据库。数据库的使用需要官方给定的**密钥**，数据库的更新方式有**手动下载**和**链接下载**。目前我个人已申请的**密钥**和**下载链接**如下：

```markdown
# 密钥
CK0b/HnKNoszm+FGt+jSoA==
# 下载链接
https://www.cz88.net/api/communityIpAuthorization/communityIpDbFile?fn=czdb&key=f07d138c-3349-3b2c-9d88-b76da90e939c
```

## 2、python代码

### 2.1 环境准备

参考官方给出的[python代码](https://github.com/tagphi/czdb_searcher_python)，将`czdb`文件复制到工作目录。

```shell
pip install msgpack
pip install pycryptodome
```

### 2.2 示例

```python
import sys
from czdb.db_searcher import DbSearcher

database_path = "/path/to/your/database.czdb"
query_type = "BTREE"
key = "YourEncryptionKey"
ip = "8.8.8.8"

db_searcher = DbSearcher(database_path, query_type, key)

try:
    region = db_searcher.search(ip)
    print("搜索结果：")
    print(region)
except Exception as e:
    print(f"An error occurred during the search: {e}")

db_searcher.close()
```

### 2.3 工具代码

代码逻辑：

1. 输入`ip`地址。
2. 判断`ip`数据库是否需要更新，更新逻辑为：
   1. 首次使用工具需要下载最新的`ip`数据库。
   2. 当前`ip`数据库下载时间是上周三之前，并且当天时间是本周的周三之后，需要更新。
   3. 否则不需要更新。
3. 返回结果。

```python
from typing import Any

import os
import sys
sys.path.append(os.path.dirname(__file__))

import shutil
import requests
import zipfile
import subprocess
import datetime as dt

from datetime import datetime, timedelta
from czdb.db_searcher import DbSearcher
from core.tools.entities.tool_entities import ToolInvokeMessage
from core.tools.tool.builtin_tool import BuiltinTool


class IpToAdressConvert(BuiltinTool):

    def _invoke(self, user_id: str, tool_parameters: dict[str, Any]) -> ToolInvokeMessage | list[ToolInvokeMessage]:
        ip = tool_parameters.get('query', '')
        api_key = self.runtime.credentials['serpapi_api_key']

        return self.create_text_message(self.ip_to_adress(ip, api_key))
    

    def ip_to_adress(self, ip, api_key):
        current_dir_path = os.path.dirname(__file__)
        dataset_dir = os.path.join(current_dir_path, 'dataset_dir')
        # 文件更新验证
        try:
            self.file_update(current_dir_path, dataset_dir)
        except Exception as e:
            print(f'An error occurred: {e}')

        database_path = os.path.join(dataset_dir, 'cz88_public_v4.czdb')
        query_type = "BTREE"
        db_searcher = DbSearcher(database_path, query_type, api_key)
        result = ''
        try:
            result = db_searcher.search(ip)
        except Exception as e:
            result = f"An error occurred during the search: {e}"

        db_searcher.close()
        return result
    

    def file_update(self, current_dir_path, dataset_dir):
        # 文件更新过
        if self.is_file_updated(dataset_dir):
            return

        # 增加权限
        self.execute_chmod_command(current_dir_path)

        exclusive_download_link = 'https://www.cz88.net/api/communityIpAuthorization/communityIpDbFile?fn=czdb&key=f07d138c-3349-3b2c-9d88-b76da90e939c'
        response = requests.get(exclusive_download_link)
        response.raise_for_status()

        zip_file_path = os.path.join(current_dir_path, 'czdb.zip')

        if os.path.exists(dataset_dir):
            self.remove_directory(dataset_dir)

        # 删除旧的czdb.zip
        if os.path.exists(zip_file_path):
            os.rmdir(zip_file_path)

        with open(zip_file_path, 'wb') as f:
            f.write(response.content)

        if not os.path.exists(dataset_dir):
            os.mkdir(dataset_dir)
        
        # 解压ZIP文件
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
            os.remove(zip_file_path)
        print(f'文件更新完成！')

    
    def is_file_updated(self, dataset_dir):
        '''
            判断文件是否在一周内更新过
            1. 官方通常情况下每周三更新一次，因此为避免更新到旧的ip库，在 周四 至 周日 进行文件更新
            2. 这周(7天内)更新过，无需更新
            3. 假如是第一次更新，没有dataset_dir
        '''
        if not os.path.exists(dataset_dir):
            return False
        
        if dt.date.today().weekday() < 3:
            return True
        
        creation_time = os.path.getctime(dataset_dir)
        creation_datetime = datetime.fromtimestamp(creation_time)
        now = datetime.now()
        delta = now - creation_datetime

        if delta > timedelta(days=7):
            return False
        return True

    def remove_directory(self, directory_path):
        try:
            # 删除目录及其内容
            shutil.rmtree(directory_path)
            print(f"Directory '{directory_path}' and its contents have been successfully removed.")
        except FileNotFoundError:
            print(f"The directory '{directory_path}' does not exist.")
        except PermissionError as e:
            print(f"Permission denied: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")


    def execute_chmod_command(self, directory_path):
        try:
            # 构造命令
            command = ['chmod', '-R', '777', directory_path]

            # 执行命令
            result = subprocess.run(command, check=True)

            # 输出结果
            print(f"Command executed successfully with return code {result.returncode}.")
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {e}")
        except FileNotFoundError:
            print(f"The directory '{directory_path}' does not exist.")
        except Exception as e:
            print(f"An error occurred: {e}")
```

# Dify项目的web页面修改

## 1、增加配置到docker-compose.yaml

找到`web`服务，添加`build`配置。`context`填写`docker-compose.yaml`相对于`web`文件夹的路径，`dockerfile`填写`web`文件夹下的`Dockerfile`路径（这里填写`Dockerfile`是因为它会自动寻找`context`目录下的`Dockerfile`文件）。

```bash
web:
    build:
      context: ../web
      dockerfile: Dockerfile
    image: langgenius/dify-web:0.14.2
    restart: always
    environment:
      CONSOLE_API_URL: ${CONSOLE_API_URL:-}
      APP_API_URL: ${APP_API_URL:-}
      SENTRY_DSN: ${WEB_SENTRY_DSN:-}
      NEXT_TELEMETRY_DISABLED: ${NEXT_TELEMETRY_DISABLED:-0}
      TEXT_GENERATION_TIMEOUT_MS: ${TEXT_GENERATION_TIMEOUT_MS:-60000}
      CSP_WHITELIST: ${CSP_WHITELIST:-}
```

## 2、问题

找到`web`文件夹下的`Dockerfile`，注释掉下面这一行，否则使用`yarn install`会失败。

```bash
# COPY yarn.lock .
```

## 3、运行

```bash
sudo docker compose up -d --build
```

**等待编译完成后即可。**
