module convnet

using mnist

# tipuri
typealias Neurons{T} Dict{Int, Array{T}}
typealias Errors{T}  Dict{Int, Array{T}}
typealias Weights{T} Dict{Int, Array{T}}
typealias Gradients{T} Dict{Int, Array{T}}

typealias NetworkConfig Array{Dict{String,Any}, 1}

type FwdComputation{T}
  batchSize::Int64
  A::Neurons{T}
  Z::Neurons{T}
end

type BackComputation{T}
  batchSize::Int64
  E::Errors{T}
  G::Gradients{T}
end

function init_weights(T::DataType, config::NetworkConfig,
                      minValue::Float64 = -0.5, maxValue::Float64 = 0.5)
  # Dictionary of weights
  weights = Weights{T}();
  # Size of inputs and number of maps from previous layer
  width = config[1]["width"];
  maps_no = 1;
  # Init weights
  for l in 2:(length(config))
    layer = config[l];
    if layer["type"] == "CONV"
      weights[l] = rand(T, layer["maps"], maps_no, layer["r"] * layer["r"] + 1);
      weights[l] = minValue + weights[l] * maxValue;
      maps_no = layer["maps"];
      width = width - layer["r"] + 1;
    elseif layer["type"] == "MAXPOOL"
      width = width / layer["r"];
    elseif layer["type"] == "FULL"
      weights[l] = rand(layer["m"], convert(Int64,maps_no * width * width + 1));
      weights[l] = minValue + weights[l] * maxValue;
      maps_no = layer["m"];
      width = 1;
    end
  end
  return weights
end

function prepare_network(T::DataType, config::NetworkConfig, batchSize)
  A = Neurons{T}();
  Z = Neurons{T}();
  E = Neurons{T}();
  G = init_weights(T, config, 0.0, 0.0);
  # Size of inputs and number of maps from previous layer
  width = config[1]["width"];
  maps_no = 1;
  # Init weights
  for l in 2:(length(config))
    layer = config[l];
    if layer["type"] == "CONV"
      width = width - layer["r"] + 1;
      maps_no = layer["maps"];
      A[l] = zeros(T, batchSize, maps_no, width, width);
      Z[l] = zeros(T, batchSize, maps_no, width, width);
      E[l] = zeros(T, batchSize, maps_no, width, width);
    elseif layer["type"] == "MAXPOOL"
      width = convert(Int64, width / layer["r"]);
      Z[l] = zeros(T, batchSize, maps_no, width, width);
      E[l] = zeros(T, batchSize, maps_no, width, width);
    elseif layer["type"] == "FULL"
      maps_no = layer["m"];
      width = 1;
      A[l] = zeros(T, batchSize, maps_no);
      Z[l] = zeros(T, batchSize, maps_no);
      E[l] = zeros(T, batchSize, maps_no);
    end
  end
  return FwdComputation(batchSize, A, Z), BackComputation(batchSize, E, G);
end

function describe_weights{T}(W::Weights{T})
  for w in W
    print("Weights on layer #", w[1], " have size ", size(w[2]));
    println(" (", prod(size(w[2])), " parameters)");
  end
end

layers = Dict{String, Any}[{"type" => "INPUTS", "width" => 28},
                           {"type" => "CONV", "r" => 5, "maps" => 6,
                            "mapping" => Bool[1 1 1 1 1 1],
                            "activation" => "sigmoid"},
                           {"type" => "MAXPOOL", "r" => 2},
                           {"type" => "CONV", "r" => 5, "maps" => 13,
                            "mapping" => Bool[1 1 0 1 1 1 1 1 0 0 1 1 1;
                                              1 1 1 0 1 1 1 1 1 0 0 1 1;
                                              1 1 1 1 0 1 1 1 1 1 0 0 1;
                                              1 1 1 1 1 0 1 1 1 1 1 0 0;
                                              1 1 1 1 1 1 0 0 1 1 1 1 0;
                                              1 0 1 1 1 1 1 0 0 1 1 1 1],
                            "activation" => "sigmoid"},
                           {"type" => "MAXPOOL", "r" => 2},
                           {"type" => "FULL", "m" => 400, "activation" => "sigmoid"},
                           {"type" => "FULL", "m" => 10, "activation" => "softmax"}
                           ];

sigmoid(x) = 1 ./ (1 .+ exp(-x));

function softmax{T}(a::Array{T,1})
  y = 1 ./ 1 .+ exp(a);
  y = y / sum(y);
end

function forward{T}(config::NetworkConfig, X::Array{T,2},
                    W::Weights{T}, nn::FwdComputation{T})
  maps_no = 1;
  width = config[1]["width"];

  nn.Z[1] = reshape(X, nn.batchSize, 1, width, width);

  for l in 2:(length(config)-1)

    println("Gigi va da bani: ", l);

    layer = config[l];

    if layer["type"] == "CONV"
      println("Gigi ", l, " este convolutiv si recalcitrant");
      new_width = width - layer["r"] + 1;
      new_maps_no = layer["maps"];

      r = layer["r"];
      for n in 1:nn.batchSize
        for m in 1:new_maps_no
          for i in 1:new_width
            for j in 1:new_width
              nn.A[l][n, m, i, j] = zero(T);
              for prev_m in 1:maps_no
                input = [one(T) reshape(nn.Z[l-1][n, prev_m, i:(i+r-1), j:(j+r-1)], 1, r * r)];
                nn.A[l][n, m, i, j] += layer["mapping"][prev_m,m] * (input * reshape(W[l][m,prev_m,:], r*r+1,1))[1];
              end
            end
          end
        end
      end
      nn.Z[l] = sigmoid(nn.A[l]);

      width = new_width;
      maps_no = new_maps_no;
    elseif layer["type"] == "MAXPOOL"
      r = layer["r"];
      new_width = convert(Int64, width / r);
      for n in 1:nn.batchSize
        for (m in 1:maps_no)
          for (i in 1:new_width)
            for (j in 1:new_width)
              nn.Z[l][n, m, i, j] =
                maximum(nn.Z[l-1][n, m, (r*(i-1)+1):(r*i), (r*(i-1)+1):(r*i)]);
            end
          end
        end
      end
      width = new_width;
    elseif layer["type"] == "FULL"
      for (n in 1:nn.batchSize)
        for (j in 1:layer["m"])
          nn.A[l][n,j] =
            ([1 reshape(nn.Z[l-1][n,1:end,1:end,1:end], 1, width * maps_no * width)]
             *
             vec(W[l][j,:]))[1];
        end
      end
      if layer["activation"] == "sigmoid"
        nn.Z[l] = sigmoid(nn.A[l]);
      elseif layer["activation"] == "softmax"
        nn.Z[l] = softmax(nn.A[l]);
      end
      width = 1;
      maps_no = layer["m"];
    end
  end
end

function backprop{T}(config::NetworkConfig, X::Array{T, 2}, L::Array{T, 2},
                     W::Weights{T}, nn::FwdComputation{T},
                     err::BackComputation{T})
  # Se calculează erorile și gradienții.
  # Ultimul strat se calculează înaintea celorlalte.
  # Într-un layer maxpul facem așa:
  #   1. calculăm eroarea (ca sumă, plm, după toate cele de pe stratul viitor)
  #   2. calculăm gradient (plm, super simplu, o înmulțire)
  #   3. punem eroarea și la elementul maxim de la convoluta de mai devreme
  # La o convolută facem după cum se scurge mai departe:
  #   1. ne-a pus maxpul erorile în erori
  #   2. calculăm doar gradienții
  #   3. gradienții să fie medie peste toate erorile puse în aceeași mapă (hartă n.r.)
  # La o full e cel mai simplu
  #   1. se adună și se înmulțesc
  # încă ceva, presupune că ultimul strat este full conectat
  
  forward(config, X, W, nn);  # forward
  # Last layer
  println("dam la iesire intai");

  l = length(config);

  layer = config[l];
  
  if layer["activation"] == "softmax" || layer["activation"] == "sigmoid"
    err.E[l][:,:] = nn.Z[l] .- L';
  else
    println("Exceptie: nu s-a implementat inca");
  end
  
  for j in 1:layer["m"]
    if config[l-1]["type"] == "FULL"
      err.G[l][j, 2:end] = sum(err.E[l][j] * nn.Z[l-1],1);
      err.G[l][j, 1] = err.E[l][j];
    elseif config[l-1]["type"] == "MAXPOOL"
      i = 1;
      err.G[l][j, i] = err.E[l][j];
      for m in 1:config[l-1]["maps"]
        for l1 in 1:config[l-1]["width"]
          for l2 in 1:config[l-1]["width"]
            i = i+1;
            err.G[l][j, i] = sum(err.E[l][j] * nn.Z[l-1][m, l1, l2]);
          end
        end
      end
    else
      println("Daca ai ajuns aici esti un chinez!");
    end
  end

  # All the other layers
  for l in (length(config)-1):-1:2
    println("facem stratul ", l);
    layer = config[l];
    if layer["type"] == "CONV"
      println("Gigi e convolut");
      # presupunem ca maxpul ne-a pus eroarea
    elseif layer["type"] == "MAXPOOL"
      # maxpul -> full
      if config[l+1]["type"] == "FULL"
        for m in 1:layer["maps"]
          width = size(nn.Z[l],3);
          for i in 1:width
            for j in 1:width
              err.E[l+1] * W[l+1][:,(m-1) * width*width + (i-1)*width + j + 1];
            end
          end
        end
      else
        #it should be conv
        for m in 1:layer["maps"]
          width = size(nn.Z[l],3);
          for i in 1:width
            for j in 1:width
              sum(err.E[l+1][,:m,:,:] * [:,:,2:])
            end
          end
        end
      end
        
      
    elseif layer["type"] == "FULL"
      println("Gigi e complet conectat");
    else
      println("Daca ai ajuns aici esti un oier!");
    end
  end
end

function test()
  d = read_mnist_data();
  w = init_weights(Float64, layers, -0.1, 0.1);
  nn, err = prepare_network(Float64, layers, 100);
  describe_weights(w);
  backprop(layers, d["train_img"][:,1:100], d["train_label"][:,1:100], w, nn, err);
end

end
