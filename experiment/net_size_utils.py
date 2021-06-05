def make2d(int_or_tuple):
    if(type(int_or_tuple) is int):
        return (int_or_tuple,int_or_tuple)
    else:
        return int_or_tuple
# Returns the dimensions of the image after performing nn.Conv2d
def dim_conv2d(size_in, kernel_size, stride=1, padding=0, dilation=1):
    kernel_size = make2d(kernel_size)
    stride = make2d(stride)
    padding = make2d(padding)
    dilation = make2d(dilation)
    height_out = int((size_in[0] + 2*padding[0] - dilation[0]*(kernel_size[0]-1) - 1)/stride[0] + 1)
    width_out = int((size_in[1] + 2*padding[1] - dilation[1]*(kernel_size[1]-1) - 1)/stride[1] + 1)
    return (height_out, width_out)

# Returns the dimensions of the image after performing nn.MaxPool2d
def dim_maxpool2d(size_in, kernel_size, stride=None, padding=0, dilation=1):
    if stride == None:
        stride = kernel_size
    # The formula is the same as for dim_conv2d
    return dim_conv2d(size_in, kernel_size, stride=stride, padding=padding, dilation=dilation)
