var express = require('express');
var app = express();

app.set("view engine", "ejs");
app.set("views", __dirname + "/");
app.set("view options", { layout: false } );

app.get('/', function(req, res) {
    console.log("==================");
    console.log(req.params);
    console.log("==================");
	res.sendFile(__dirname + '/index.html');					  
});

app.listen(8080);
console.log('listening on port 8080...');