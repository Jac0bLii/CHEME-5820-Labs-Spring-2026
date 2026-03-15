function build_cbow_pairs(sentences::Array{String,1}, vocabulary::Dict{String,Int64}; window_size::Int64=2)
    vocab_size = length(vocabulary)
    training_pairs = []
    for sentence in sentences
        words = split(lowercase(sentence))
        n = length(words)
        for i in 1:n
            target_word = get(vocabulary, words[i], get(vocabulary, "<unk>", 0))
            if target_word == 0
                continue
            end
            target = zeros(Float64, vocab_size)
            target[target_word] = 1.0
            context_sum = zeros(Float64, vocab_size)
            count = 0
            for j in max(1, i-window_size):min(n, i+window_size)
                if j == i
                    continue
                end
                ctx_word = get(vocabulary, words[j], get(vocabulary, "<unk>", 0))
                if ctx_word > 0
                    context_sum[ctx_word] += 1.0
                    count += 1
                end
            end
            if count > 0
                push!(training_pairs, (context_sum, target))
            end
        end
    end
    return training_pairs
end

function train_cbow(training_pairs, vocab_size::Int64; d_h::Int64=5, eta::Float64=0.01, num_epochs::Int64=500)
    W1 = randn(d_h, vocab_size) * 0.01
    W2 = randn(vocab_size, d_h) * 0.01
    loss_history = zeros(Float64, num_epochs)
    for epoch in 1:num_epochs
        total_loss = 0.0
        for (x, y) in training_pairs
            h = W1 * x
            u = W2 * h
            exp_u = exp.(u .- maximum(u))
            y_hat = exp_u ./ sum(exp_u)
            loss = -sum(y .* log.(y_hat .+ 1e-10))
            total_loss += loss
            delta_u = y_hat .- y
            dW2 = delta_u * h'
            delta_h = W2' * delta_u
            dW1 = delta_h * x'
            W2 .-= eta .* dW2
            W1 .-= eta .* dW1
        end
        loss_history[epoch] = total_loss / length(training_pairs)
    end
    return W1, W2, loss_history
end

function nearest_neighbors(W1::Matrix{Float64}, vocabulary::Dict{String,Int64},
                            word::String; top_k::Int=5)
    idx = get(vocabulary, word, 0)
    if idx == 0
        return Tuple{String,Float64}[]
    end
    v = W1[:, idx]
    v_norm = norm(v)
    inv_vocab = Dict{Int64,String}(v => k for (k, v) in vocabulary)
    vocab_size = length(vocabulary)
    sims = fill(-Inf, vocab_size)
    for i in 1:vocab_size
        if i == idx
            continue
        end
        vi = W1[:, i]
        ni = norm(vi)
        sims[i] = (v_norm > 1e-10 && ni > 1e-10) ? dot(v, vi) / (v_norm * ni) : 0.0
    end
    top_indices = sortperm(sims, rev=true)[1:min(top_k, vocab_size)]
    return [(inv_vocab[i], round(sims[i]; digits=4)) for i in top_indices]
end

function solve_analogy(W1::Matrix{Float64}, vocabulary::Dict{String,Int64},
                       word_a::String, word_b::String, word_c::String;
                       top_k::Int=5, exclude_inputs::Bool=true)
    for w in [word_a, word_b, word_c]
        if !haskey(vocabulary, w)
            return Tuple{String,Float64}[]
        end
    end
    v_a = W1[:, vocabulary[word_a]]
    v_b = W1[:, vocabulary[word_b]]
    v_c = W1[:, vocabulary[word_c]]
    v_target = v_b .- v_a .+ v_c
    v_target_norm = norm(v_target)
    inv_vocab = Dict{Int64,String}(v => k for (k, v) in vocabulary)
    vocab_size = length(vocabulary)
    sims = zeros(Float64, vocab_size)
    for i in 1:vocab_size
        vi = W1[:, i]
        ni = norm(vi)
        sims[i] = (v_target_norm > 1e-10 && ni > 1e-10) ? dot(v_target, vi) / (v_target_norm * ni) : 0.0
    end
    if exclude_inputs
        for w in [word_a, word_b, word_c]
            sims[vocabulary[w]] = -Inf
        end
    end
    top_indices = sortperm(sims, rev=true)[1:min(top_k, vocab_size)]
    return [(inv_vocab[i], round(sims[i]; digits=4)) for i in top_indices]
end
