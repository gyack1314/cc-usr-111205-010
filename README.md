# 安置到达时延的联合分位回归与不交叉域

这是独立的防汛演练资料Python项目。`data/`保存虚构人数、观测、时序或方案，`response_data.io`提供严格数值读取与来源字节SHA-256，`response_data.domain`给出已有原始摘要，`response_data.cli`输出JSON。字段、量纲与分析口径见[资料约定](docs/data-contract.md)，联合LP、不交叉保证范围与前推验证见[方法说明](docs/method.md)。这些数据用于离线复盘，不连接真实人员、传感器或指挥系统。

Python 3.11及以上即可直接使用，无第三方Python依赖。在项目根目录执行：

```sh
python -m unittest discover -s tests -v
python -m compileall -q response_data tests
python -m response_data.cli inspect data/normal.json
python -m response_data.cli check-domain data/normal.json --point '[1.5,1.5]'
python -m response_data.cli fit data/normal.json
```

`inspect`显示原始资料概况，不覆盖源文件。`check-domain`在原协变量单位给出凸包成员判定与非负重心见证。`fit`运行完整流水线：按完整过程时间前推验证选择lambda、全开发窗重拟合冻结，并输出逐行可重算的残差与损失。联合分位在调用者给定的凸包**每个公开顶点**上约束相邻分位差非负，线性保证覆盖整个凸包且仅限该凸包；域外预测只给无保证标识。

`invalid-`前缀资料为不合法输入，其余变体使用相同字段保留不同业务边界：`external-outside-domain`的外期点在凸包外，`incomplete-covariate`含缺失字段行（不参与拟合）。数值null须按字段语义区别于0，不在读取时自动填补。

Dockerfile固定基础镜像，可用`docker build -t response-data .`及`docker run --rm response-data`读取资料；`docker run --rm response-data python -m unittest discover -s tests -v`执行已有检查。运行所需资料全部在本地，无需登录或实时来源。
