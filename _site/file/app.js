const express = require('express');
const bodyParser = require('body-parser');

const app = express();

// 配置body-parser中间件
app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());

// 处理POST请求
app.post('/save-data', (req, res) => {
  const data = req.body;
  console.log('Received data:', data);
  // 将数据保存到服务器上，例如使用文件系统或数据库
  // ...
  res.send('Data received and saved!');
});

// 启动服务器
const port = 3000;
app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});