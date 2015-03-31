#using PyCall;
#using PyPlot;

module mnist

export read_mnist_data, show_some_imgs

# Read images from file
function read_mnist_img_file(path::String, correct_magic_number::Int32)
    img_file = open(path, "r")
    magic_number = bswap(read(img_file, Int32));
    @assert magic_number == correct_magic_number
    n = int64(bswap(read(img_file, Int32)));
    rows = int64(bswap(read(img_file, Int32)));
    columns = int64(bswap(read(img_file, Int32)));
    data = float64(read(img_file, Uint8, (rows * columns, n))) ./ 255.0
    close(img_file);
    return data;
end

# Read labels
function read_mnist_label_file(path::String, correct_magic_number::Int32)
    label_file = open(path, "r")
    magic_number = bswap(read(label_file, Int32));
    @assert magic_number == correct_magic_number
    n = int64(bswap(read(label_file, Int32)));
    data = read(label_file, Uint8, (n))
    targets = zeros(10, n)
    for i = 1:n
        targets[data[i]+1, i] = 1
    end
    close(label_file);
    return targets;
end

function read_mnist_data()
    # Read images
    train_img = read_mnist_img_file("data/train-images.idx3-ubyte", int32(2051))
    test_img = read_mnist_img_file("data/t10k-images.idx3-ubyte", int32(2051))
    # Read labels
    train_label = read_mnist_label_file("data/train-labels.idx1-ubyte",
                                        int32(2049))
    test_label = read_mnist_label_file("data/t10k-labels.idx1-ubyte",
                                       int32(2049))
    return ["train_img" => train_img,
            "test_img" => test_img,
            "train_label" => train_label,
            "test_label" => test_label]
end

function show_some_imgs(imgs, labels, rows::Integer, cols::Integer)
    d = zeros(rows, cols)

    index1 = ceil(rand()*size(imgs,2))
    row = permutedims(reshape(imgs[:,index1], 28, 28), [2,1])
    d[1,1] = find(labels[:,index1])[1]
    for j = 2:cols
        index1 = ceil(rand()*size(imgs,2))
        row = hcat(row, permutedims(reshape(imgs[:,index1], 28, 28), [2,1]))
        d[1,j] = find(labels[:,index1])[1]
    end

    m = row

    for i = 2:rows
        index1 = ceil(rand()*size(imgs,2))
        row = permutedims(reshape(imgs[:,index1], 28, 28), [2,1])
        d[i,1] = find(labels[:,index1])[1]
        for j = 2:cols
            index1 = ceil(rand()*size(imgs,2))
            row = hcat(row, permutedims(reshape(imgs[:,index1], 28, 28), [2,1]))
            d[i,j] = find(labels[:,index1])[1]
        end
        m = vcat(m, row)
    end
    println(int(d .- 1))
    imshow(m)
end

end
