# Tudor Berariu <tudor.berariu@gmail.com>, 2015

module Bullseye

export bullseye

function bullseye{T<:FloatingPoint}(N::Integer, D::Integer,
                                    radii::Array{T,1}, widths::Array{T,1})
    if length(radii) != length(widths)
        error("Parameters radii and widths must have same length");
    end
    const M = length(radii);
    data = rand(T, D, N);
    labels = rand(1:M, N);
    for i = 1:N
        len = radii[labels[i]] + randn() * widths[labels[i]] * 0.5;
        angle = rand(D-1)*2*pi;
        sins = one(T);
        for j = 1:D-1
            data[j,i] = len * sins * cos(angle[j]);
            sins *= sin(angle[j]);
        end
        data[D,i] = len * sins;
    end
    return data, labels;
end

function bullseye2{T<:FloatingPoint}(N::Integer, D::Integer,
                                    radii::Array{T,1}, widths::Array{T,1})
    if length(radii) != length(widths)
        error("Parameters radii and widths must have same length");
    end
    const M = length(radii);
    data = rand(T, D, N);
    labels = rand(1:M, N);
    len = radii[labels] + randn(N) .* widths[labels] * 0.5;
    angle = rand(N, D-1) * 2 * pi;
    for i = 1:N
        len = radii[labels[i]] + randn() * widths[labels[i]] * 0.5;
        angle = rand(D-1)*2*pi;
        sins = one(T);
        for j = 1:D-1
            data[j,i] = len * sins * cos(angle[j]);
            sins *= sin(angle[j]);
        end
        data[D,i] = len * sins;
    end
    return data, labels;
end

end
