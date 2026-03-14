function build_bow_matrix(sentences::Array{String,1}, vocabulary::Dict{String,Int64})::Array{Int64,2}
    num_sentences = length(sentences)
    vocab_size = length(vocabulary)
    bow_matrix = zeros(Int64, num_sentences, vocab_size)
    for (i, sentence) in enumerate(sentences)
        words = split(lowercase(sentence))
        augmented = ["<bos>"; words; "<eos>"]
        for word in augmented
            idx = get(vocabulary, word, get(vocabulary, "<unk>", 0))
            if idx > 0
                bow_matrix[i, idx] += 1
            end
        end
    end
    return bow_matrix
end

function hashing_vectorizer(features::Array{String,1}; length::Int64=10)::Array{Int64,1}
    v = zeros(Int64, length)
    for f in features
        idx = mod1(abs(hash(f)), length)
        v[idx] += 1
    end
    return v
end
