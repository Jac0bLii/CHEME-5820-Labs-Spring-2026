function build_cooccurrence_matrix(sentences::Array{String,1}, vocabulary::Dict{String,Int64}; window_size::Int64=2)::Array{Int64,2}
    vocab_size = length(vocabulary)
    cooccurrence = zeros(Int64, vocab_size, vocab_size)
    for sentence in sentences
        words = split(lowercase(sentence))
        for (i, word) in enumerate(words)
            w_idx = get(vocabulary, word, get(vocabulary, "<unk>", 0))
            if w_idx == 0
                continue
            end
            lo = max(1, i - window_size)
            hi = min(length(words), i + window_size)
            for j in lo:hi
                if j == i
                    continue
                end
                ctx = get(vocabulary, words[j], get(vocabulary, "<unk>", 0))
                if ctx > 0
                    cooccurrence[w_idx, ctx] += 1
                end
            end
        end
    end
    return cooccurrence
end

function build_pmi_matrices(cooccurrence_matrix::Array{Int64,2})
    vocab_size = size(cooccurrence_matrix, 1)
    total = sum(cooccurrence_matrix)
    if total == 0
        return zeros(Float64, vocab_size, vocab_size), zeros(Float64, vocab_size, vocab_size)
    end
    joint_prob = cooccurrence_matrix ./ total
    marginal_w = sum(joint_prob, dims=2)
    marginal_c = sum(joint_prob, dims=1)
    pmi_matrix = fill(-Inf, vocab_size, vocab_size)
    for i in 1:vocab_size
        for j in 1:vocab_size
            if joint_prob[i,j] > 0 && marginal_w[i] > 0 && marginal_c[j] > 0
                pmi_matrix[i,j] = log2(joint_prob[i,j] / (marginal_w[i] * marginal_c[j]))
            end
        end
    end
    ppmi_matrix = max.(pmi_matrix, 0.0)
    return pmi_matrix, ppmi_matrix
end
