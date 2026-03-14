function build_tf_matrix(bow_matrix::Array{Int64,2})::Array{Float64,2}
    num_sentences, vocab_size = size(bow_matrix)
    tf_matrix = zeros(Float64, num_sentences, vocab_size)
    for i in 1:num_sentences
        total = sum(bow_matrix[i, :])
        if total > 0
            tf_matrix[i, :] = bow_matrix[i, :] ./ total
        end
    end
    return tf_matrix
end

function build_idf_dictionary(bow_matrix::Array{Int64,2}, vocabulary::Dict{String,Int64}, num_sentences::Int64)::Dict{String,Float64}
    idf_dict = Dict{String, Float64}()
    inverse_vocab = Dict{Int64, String}(v => k for (k, v) in vocabulary)
    vocab_size = size(bow_matrix, 2)
    for j in 1:vocab_size
        df = sum(bow_matrix[:, j] .> 0)
        word = get(inverse_vocab, j, "<unk>")
        idf_dict[word] = log((num_sentences + 1) / (df + 1))
    end
    return idf_dict
end

function build_tfidf_matrix(tf_matrix::Array{Float64,2}, idf_dict::Dict{String,Float64}, inverse_vocabulary::Dict{Int64,String})::Array{Float64,2}
    num_sentences, vocab_size = size(tf_matrix)
    tfidf_matrix = zeros(Float64, num_sentences, vocab_size)
    for j in 1:vocab_size
        word = get(inverse_vocabulary, j, "<unk>")
        idf = get(idf_dict, word, 0.0)
        tfidf_matrix[:, j] = tf_matrix[:, j] .* idf
    end
    return tfidf_matrix
end
