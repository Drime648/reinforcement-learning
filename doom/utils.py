def get_input_shape(Image,Filter,Stride):
    layer1 = math.ceil(((Image - Filter + 1) / Stride))
    
    o1 = math.ceil((layer1 / Stride))
    
    layer2 = math.ceil(((o1 - Filter + 1) / Stride))
    
    o2 = math.ceil((layer2 / Stride))
    
    layer3 = math.ceil(((o2 - Filter + 1) / Stride))
    
    o3 = math.ceil((layer3  / Stride))

    return int(o3)