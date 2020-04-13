'use strict';
var pythonBridge = import('../node_modules/python-bridge');

console.log(pythonBridge);

let python = pythonBridge({
    python: 'python3'
});

python.ex`import ml_nn.py`