reactionrnn_build <- function(weights_path, num_classes, maxlen=140) {
  K <- backend()

  input <- layer_input(shape = c(maxlen), name='input')
  embedded <- input %>%
                layer_embedding(num_classes, 100, input_length=maxlen,
                                name='embedding')
  rnn <- embedded %>%
            layer_gru(256, name='rnn')

  output <- rnn %>%
              layer_dense(5, name='output', activation=function(x) K$relu(x) / K$sum(K$relu(x)))

  model = keras_model(inputs = input, outputs = output)
  model %>% load_model_weights_hdf5(weights_path, by_name=TRUE)
  model %>% compile(optimizer = 'nadam',loss = 'mse')

  return(model)
}
