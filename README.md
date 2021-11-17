Big-torch library
=================


### Usage:

Big-torch library installation:
```bash
pip install -i https://test.pypi.org/simple/ bigtorch-denissimo
```

Container run:

MNIST example:
```bash
docker build --pull -f "containers/mnist/Dockerfile.prod" -t bigtorch_mnist:demo "containers/mnist"
docker run -ti -rm bigtorch_mnist:demo
```

RNN example:
```bash
docker build --pull -f "containers/rnn/Dockerfile.prod" -t bigtorch_rnn:demo "containers/rnn"
docker run -ti -rm bigtorch_rnn:demo
```

### Context:

#### 1) Big-troch lib

##### It has the following structure:

    big_torch:
    ├── core                    # Kernel structures like computaional graph and so on
    ├── utils                   # Mix of different utilities I'm currently unable to gather in one logical module
    ├── layers                  # All possible layers that can form my neural network
    ├── model                   # Model builder
    ├── train                   # Is represented by OptimizationFabric which can easily compile complex training algorithms
    ├── preprocessing           # Some useful utilities you may need during your research
    └── remote_client.py        # Allows you to launch models by special configuration files


##### Code example:

Case 1: Sequental model
```python
from big_torch import models, train, layers, preprocessing


net = models.sequental.Sequental()

net.add_layer(layers.linear.LinearLayer((784, 100), b_initial=0, kernel_initializer='xavier_normal'))
net.add_layer(layers.activations.Tanh((100, 100)))
net.add_layer(layers.linear.LinearLayer((100, 100) , b_initial=0, kernel_initializer='xavier_normal'))
net.add_layer(layers.activations.Tanh((100, 100)))
net.add_layer(layers.linear.LinearLayer((100, 10) , b_initial=0, kernel_initializer='xavier_normal'))
net.add_layer(layers.activations.Softmax((10, 10)))
net.set_loss(layers.loss.CrossEntropy((10, 1)))
net.compile()

optimizer = train.fabric.OptimizatonFabric(
    generator=train.frame_generators.StochasticBatchGenerator, 
    optimizer=train.optimizers.GradientDecent,
    optimizer_cfg={'eta': 0.1},
    generator_cfg={'batch_size': 1000}, 
    callbacks=[
        train.callbacks.EpochInfo(
            validate=True, 
            metrics={
                'accuracy': preprocessing.cross_validation.class_accuracy
            })
    ]
)

optimizer.train(net, t_x, t_y, x_val=v_x, y_val=v_y, min_eps=0.0000005, max_iter=200)
```

Case 2: GraphModel
```python
from big_torch import models, train, layers, preprocessing


input = layers.variables.Placeholder(shape=(784))()

l1 = layers.linear.LinearLayer((784, 100), b_initial=0, kernel_initializer='xavier_normal')(input)
a1 = layers.activations.Tanh((100, 100))(l1)
l2 = layers.linear.LinearLayer((100, 100) , b_initial=0, kernel_initializer='xavier_normal')(a1)
a2 = layers.activations.Tanh((100, 100))(l2)
l3 = layers.linear.LinearLayer((100, 10) , b_initial=0, kernel_initializer='xavier_normal')(a2)

o1 = layers.activations.Softmax((10, 10))(l3)
loss = layers.loss.CrossEntropy((10,1))(o1)

net = models.GraphModel(inputs=[input], outputs=[o1])
net.compile(loss)

optimizer = train.fabric.OptimizatonFabric(
    generator=train.frame_generators.StochasticBatchGenerator, 
    optimizer=train.optimizers.GradientDecent,
    optimizer_cfg={'eta': 0.1},
    generator_cfg={'batch_size': 1000}, 
    callbacks=[
        train.callbacks.EpochInfo(
            validate=True, 
            metrics={
                'accuracy': preprocessing.cross_validation.class_accuracy
            })
    ]
)

optimizer.train(net, t_x, t_y, x_val=v_x, y_val=v_y, min_eps=0.0000005, max_iter=200)
```

#### 2) Container:

This folder will provide you easy to launch docker container (Dockerfile.prod). You'll just need to specify model structure and credentials to neptune.ai in ***config*** folder. 
Note that Dockerfile.build is used in my CI and you don't need to run it manualy. 

> Results you can see on neptune tracker. [See.](https://app.neptune.ai/denissimo/MNIST-Big-Torch/)
