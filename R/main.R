#' janus
#'
#' @param data A data frame including at least three features: rating actor, rated item and rating value.
#' @param rating_label String. Single label for the feature containing the rating values.
#' @param rater_label String. Single label for the feature containing the rating actors.
#' @param rated_label String. Single label for the feature containing the rated items.
#' @param task String. Available options are: "regr", for regression (when the rating value is numeric); "classif", for classification (when the rating value is a class or a factor).
#' @param skip_shortcut Logical. Option to add a skip shortcut to improve network performance in case of many layers. Default: FALSE.
#' @param rater_embedding_size Integer. Output dimension for embedding the rating actors. Default: coarse-to-fine search (8 to 32).
#' @param rated_embedding_size Integer. Output dimension for embedding the rated items. Default: coarse-to-fine search (8 to 32).
#' @param layers Positive integer. Number of layers for DNN. Default: coarse-to-fine search (1 to 5).
#' @param activations String. String vector with the activation functions for each layer. Default: coarse-to-fine search ("elu", "selu", "relu", "sigmoid", "softmax", "softplus", "softsign", "tanh", "linear", "leaky_relu", "parametric_relu", "thresholded_relu", "swish", "gelu", "mish", "bent").
#' @param nodes Positive integer. Integer vector with nodes for each layer. Default: coarse-to-fine search (8 to 512).
#' @param regularization_L1 Positive numeric. Value for L1 regularization of loss function. Default: coarse-to-fine search (0 to 100).
#' @param regularization_L2 Positive numeric. Value for L2 regularization of loss function. Default: coarse-to-fine search (0 to 100).
#' @param dropout Positive numeric. Value for dropout parameter at each layer (bounded between 0 and 1). Default: coarse-to-fine search (0 to 1).
#' @param batch_size Positive integer. Maximum batch size for training. Default: 64.
#' @param epochs Positive integer. Maximum number of forward and backward propagation. Default: 10.
#' @param optimizer String. Standard Tensorflow/Keras Optimization methods are available. Default: coarse-to-fine search ("adam", "sgd", "adamax", "adadelta", "adagrad", "nadam", "rmsprop").
#' @param opt_metric String. Error metric to track for the coarse-to-fine optimization. Different options: for regression, "rmse", "mae", "mdae", "mape", "smape", "rae", "rrse"; for classification, "bac", "avs", "avp", "avf", "kend", "ndcg".
#' @param folds Positive integer. Number of folds for repeated cross-validation. Default: 3.
#' @param reps Positive integer. Number of repetitions for repeated cross-validation. Default: 1.
#' @param holdout Positive numeric. Percentage of cases for holdout validation. Default: 0.1.
#' @param n_steps Positive integer. Number of phases for the coarse-to-fine optimization process (minimum 2). Default: 3.
#' @param n_samp Positive integer. Number of sampled models per coarse-to-fine phase. Default: 10.
#' @param offset Positive numeric. Percentage of expansion of numeric boundaries during the coarse-to-fine optimization. Default: 0.
#' @param n_top Positive integer. Number of candidates selected during the coarse-to-fine phase. Default: 3.
#' @param seed Positive integer. Seed value to control random processes. Default: 42.
#' @param verbose Printing specific messages. Default: TRUE.
#'
#' @author Giancarlo Vercellino \email{giancarlo.vercellino@gmail.com}
#'
#' @return This function returns a list including:
#' \itemize{
#' \item pipeline:
#' \item model:
#' \itemize{
#' \item configuration: DNN hyper-parameters (layers, activations, regularization_L1, regularization_L2, nodes, dropout)
#' \item model: Keras standard model description
#' \item recommend: function to use to recommend on rating actors
#' \item plot: Keras standard history plot
#' \item training_metrics: tracking of opt_metric across folds and repetitions
#' \item test_frame: testing set with the related predictions, including
#' \item testing_metrics: summary statistics for testing
#' }
#' \item time_log
#' }
#'
#' @export
#'
#' @import keras
#' @import tensorflow
#' @importFrom dplyr %>%
#' @import purrr
#' @import forcats
#' @import tictoc
#' @import readr
#' @import ggplot2
#' @importFrom narray subset
#' @import forcats
#' @import readr
#' @import RcppAlgos
#' @import Rmpfr
#' @import ggplot2
#' @import Metrics
#' @import StatRank
#' @importFrom hash hash invert keys values
#' @importFrom stats cor predict runif sd
#' @importFrom reticulate py_set_seed
#' @importFrom lubridate seconds_to_period
#'

######


janus <- function(data, rating_label, rater_label, rated_label, task, skip_shortcut = FALSE, rater_embedding_size = c(8, 32), rated_embedding_size = c(8, 32),
                           layers = c(1, 5), activations = c("elu", "selu", "relu", "sigmoid", "softmax", "softplus", "softsign", "tanh", "linear", "leaky_relu", "parametric_relu", "thresholded_relu", "swish", "gelu", "mish", "bent"),
                           nodes = c(8, 512), regularization_L1 = c(0, 100), regularization_L2 = c(0, 100), dropout = c(0, 1),
                           batch_size = 64, epochs = 10, optimizer = c("adam", "sgd", "adamax", "adadelta", "adagrad", "nadam", "rmsprop"),
                           opt_metric = "bac", folds = 3, reps = 1, holdout = 0.1, n_steps = 3, n_samp = 10, offset = 0, n_top = 3, seed =999, verbose = TRUE)

{
  tic.clearlog()
  tic("coarse to fine")

  act <- sample(activations, n_samp, replace=TRUE)
  opt <- sample(optimizer, n_samp, replace=TRUE)
  rtr <- round(runif(n_samp, min(rater_embedding_size), max(rater_embedding_size)))
  rtd <- round(runif(n_samp, min(rated_embedding_size), max(rated_embedding_size)))
  lyr <- round(runif(n_samp, min(layers), max(layers)))
  nod <- round(runif(n_samp, min(nodes), max(nodes)))
  rl1 <- runif(n_samp, min(regularization_L1), max(regularization_L1))
  rl2 <- runif(n_samp, min(regularization_L2), max(regularization_L2))
  drp <- runif(n_samp, min(dropout), max(dropout))

  exploration_list <- list()
  performance_list <- list()

  if(n_top > n_samp){n_top <- n_samp; message("setting n_top to n_samp")}

  class_metrics <- c("bac", "avs", "avp", "avf", "ndcg", "kend")
  regr_metrics <- c("rmse", "mae", "mdae", "mape", "smape", "rae", "rrse")

  if(task == "classif" && (is.null(opt_metric) | !(opt_metric %in% class_metrics))){opt_metric <- "bac"}
  if(task == "regr" && (is.null(opt_metric) | !(opt_metric %in% regr_metrics))){opt_metric <- "mae"}

  for(s in 1:n_steps)
  {
    if(verbose){cat("step ", s,"\n")}
    param_space <- data.frame(act, opt, rtr, rtd, lyr, nod, rl1, rl2, drp)

    exploration <- pmap(param_space, ~ engine(data = data, rating_label = rating_label, rater_label = rater_label, rated_label = rated_label, task = task,
                                                      rater_embedding_size = ..3, rated_embedding_size = ..4, layers = ..5, activations = ..1, nodes = ..6, optimizer = ..2,
                                                      regularization_L1 = ..7, regularization_L2 = ..8, dropout = ..9, skip_shortcut = skip_shortcut,
                                                      batch_size = batch_size, epochs = epochs, folds = folds, reps = reps, holdout = holdout, seed = seed))

    perf <- map_dbl(exploration, ~ .x$test_metrics[opt_metric])

    performances<- data.frame(param_space, perf)
    colnames(performances) <- c("activation", "optimizer", "rater_embedding_size", "rated_embedding_size", "layers", "nodes", "regularization_L1", "regularization_L2", "dropout",  opt_metric)
    rownames(performances) <- NULL

    best_index <- as.numeric(rownames(performances[order(performances[, opt_metric], decreasing = ifelse(task == "classif", TRUE, FALSE)),][1:n_top,]))

    act <- sample(act[best_index], n_samp, replace=TRUE)
    opt <- sample(opt[best_index], n_samp, replace=TRUE)
    rtr <- floor(runif(n_samp, min(rtr[best_index])*(1-offset), max(rtr[best_index])*(1+offset)))
    rtd <- round(runif(n_samp, min(rtd[best_index])*(1-offset), max(rtd[best_index])*(1+offset)))
    lyr <- round(runif(n_samp, min(lyr[best_index])*(1-offset), max(lyr[best_index])*(1+offset)))
    nod <- round(runif(n_samp, min(nod[best_index])*(1-offset), max(nod[best_index])*(1+offset)))
    rl1 <- runif(n_samp, min(rl1[best_index])*(1-offset), max(rl1[best_index])*(1+offset))
    rl2 <- runif(n_samp, min(rl2[best_index])*(1-offset), max(rl2[best_index])*(1+offset))
    drp <- runif(n_samp, min(drp[best_index])*(1-offset), max(drp[best_index])*(1+offset))

    rtr[rtr > max(rater_embedding_size)] <- max(rater_embedding_size)
    rtd[rtd > max(rated_embedding_size)] <- max(rated_embedding_size)
    lyr[lyr > max(layers)] <- max(layers)
    nod[nod > max(nodes)] <- max(nodes)
    rl1[rl1 > max(regularization_L1)] <- max(regularization_L1)
    rl2[rl2 > max(regularization_L2)] <- max(regularization_L2)
    drp[drp > max(dropout)] <- max(dropout)

    rtr[rtr < min(rater_embedding_size)] <- min(rater_embedding_size)
    rtd[rtd < min(rated_embedding_size)] <- min(rated_embedding_size)
    lyr[lyr < min(layers)] <- min(layers)
    nod[nod < min(nodes)] <- min(nodes)
    rl1[rl1 < min(regularization_L1)] <- min(regularization_L1)
    rl2[rl2 < min(regularization_L2)] <- min(regularization_L2)
    drp[drp < min(dropout)] <- min(dropout)

    exploration_list[[s]] <- exploration
    performance_list[[s]] <- performances
  }


  pipeline <- cbind(step=base::rep(1:n_steps, each=n_samp), Reduce(rbind, performance_list))
  if(task=="regr"){best_index <- which.min(pipeline[, opt_metric])}
  if(task=="classif"){best_index <- which.max(pipeline[, opt_metric])}

  model <-flatten(exploration_list)[[best_index]]
  best_par <- pipeline[best_index, ]
  model$configuration <- best_par
  best_model <- model[-c(3, 5, 10, 11)]

  best_model$training_metrics <- best_model$training_metrics[1:2]

  funct_parameters <- match.call()

  toc(log = TRUE)
  time_log<-tail(seconds_to_period(round(parse_number(unlist(tic.log())), 0)), 1)

  outcome <- list(pipeline = pipeline, best_model = best_model, time_log = time_log)

  return(outcome)
}

###
engine <-function(data, rating_label, rater_label, rated_label, task, positive=NULL,
                          skip_shortcut = FALSE, rater_embedding_size = 10, rated_embedding_size = 10, folds=3, reps=1, holdout=0.1,
                          layers = 1, activations = "relu", regularization_L1 = 0, regularization_L2 = 0, nodes = 32, dropout = 0,
                          span=0.2, min_delta=0, batch_size=32, epochs=50,
                          output_activation=NULL, optimizer = "Adam", loss = NULL, metrics = NULL,
                          seed = 999, reproducibility = FALSE, verbose = 0)

{
  #reticulate::use_condaenv(condaenv = "reticulate", required = TRUE)

  ###REPRODUCIBILITY
  if(reproducibility == TRUE)
  {
    set.seed(seed)
    reticulate::py_set_seed(seed = seed, disable_hash_randomization = TRUE)
    keras::use_session_with_seed(seed = seed, disable_gpu = TRUE, disable_parallel_cpu = TRUE)
  }

  config <- tensorflow::tf$compat$v1$ConfigProto(gpu_options = list(allow_growth = TRUE))###intra_op_parallelism_threads=1L, inter_op_parallelism_threads=1L,
  sess <- tensorflow::tf$compat$v1$Session(config = config)

  tensorflow::tf$get_logger()$setLevel("ERROR")

  tic.clearlog()
  tic("time")

  ####SUPPORT
  bac <- function(actual, predicted)
  {
    bac <- mean(mapply(function(x) (sum((predicted == x) & (actual == x))/sum(actual == x) + sum((predicted != x) & (actual != x))/sum(actual != x))/2, x = unique(actual)), na.rm=TRUE)
    return(bac)
  }

  avf <- function(actual, predicted)
  {
    avf <- mean(mapply(function(x) sum((predicted == x) & (actual == x))/sum((predicted == x) | (actual == x)), x = unique(actual)), na.rm=TRUE)
    return(avf)
  }

  avp <- function(actual, predicted)
  {
    avp <- mean(mapply(function(x) sum((predicted == x) & (actual == x))/sum((predicted == x)), x = unique(actual)), na.rm=TRUE)
    return(avp)
  }

  avs <- function(actual, predicted)
  {
    avs <- mean(mapply(function(x) sum((predicted == x) & (actual == x))/sum((actual == x)), x = unique(actual)), na.rm=TRUE)
    return(avs)
  }

  ###CHECK FOR DATAFRAME
  if(!is.data.frame(data)){stop("need a dataframe, check out data")}
  if(anyNA(data)){stop("missing values in data")}
  data[, rater_label] <-  as.character(unlist(data[, rater_label]))
  data[, rated_label] <-  as.character(unlist(data[, rated_label]))

  rater_level <- unique(data[, rater_label, drop=TRUE])
  rater_n <- length(rater_level)
  rater_dict <- hash(c(sort(rater_level)), 1:rater_n)
  inv_rater <- invert(rater_dict)

  rated_level <- unique(data[, rated_label, drop=TRUE])
  rated_n <- length(rated_level)
  rated_dict <- hash(c(sort(rated_level)), 1:rated_n)
  inv_rated <- invert(rated_dict)

  ####Y_DATA PREPARATION FOR CLASSIF TASK
  class_names <- NULL
  if(task=="classif")
  {
    rating_factor <- factor(as.character(unlist(data[, rating_label])))
    class_names <- levels(rating_factor)
    class_num <- length(class_names)
    if(class_num < 2){stop("need at least two levels for classification")}
    rating_data <- as.data.frame(to_categorical(as.numeric(rating_factor)-1, num_classes = class_num))
    colnames(rating_data) <- paste0("level_", class_names)
  }

  if(task=="regr")
  {
    rating_data <- data[, rating_label, drop=FALSE]
  }

  ###SWITCH FROM DATAFRAME TO ARRAY
  rater_array <- data.matrix(values(rater_dict, data[, rater_label, drop=TRUE]))
  colnames(rater_array) <- rater_label
  rated_array <- data.matrix(values(rated_dict, data[, rated_label, drop=TRUE]))
  colnames(rated_array) <- rated_label
  rating_array <- data.matrix(rating_data)

  n_rows <- dim(data)[1]
  n_cols <- dim(data)[2]

  set.seed(seed)
  test_index <- sample(n_rows, ceiling(holdout*n_rows))
  train_index <- setdiff(c(1:n_rows), test_index)

  rater_train <- subset(rater_array, index=train_index, along=1)
  rated_train <- subset(rated_array, index=train_index, along=1)
  rating_train <- subset(rating_array, index=train_index, along=1)
  n_row_train <- dim(rater_train)[1]

  rater_test <- subset(rater_array, index=test_index, along=1)
  rated_test <- subset(rated_array, index=test_index, along=1)
  rating_test <- subset(rating_array, index=test_index, along=1)

  train_loss<-c()
  train_metric<-c()
  val_loss<-c()
  val_metric<-c()
  test_loss<-c()
  test_metric<-c()

  ###DESIGN OF A SINGLE  NETWORK

  if(length(activations)<layers){activations <- replicate(layers, activations[1])}
  if(length(regularization_L1)<layers){regularization_L1 <- replicate(layers, regularization_L1[1])}
  if(length(regularization_L2)<layers){regularization_L2 <- replicate(layers, regularization_L2[1])}
  if(length(nodes)<layers){nodes <- replicate(layers, nodes[1])}
  if(length(dropout)<layers){dropout <- replicate(layers, dropout[1])}

  configuration<-data.frame(layers = NA, activations = NA, regularization_L1 = NA, regularization_L2 = NA, nodes = NA, dropout = NA)
  configuration$layers <- layers
  configuration$activations <- list(activations)
  configuration$regularization_L1 <- list(regularization_L1)
  configuration$regularization_L2 <- list(regularization_L2)
  configuration$nodes <- list(nodes)
  configuration$dropout <- list(dropout)

  ###CREATION OF KERAS NEURAL NET MODELS
  rater_tensor <- layer_input(shape= dim(rater_array)[-1])
  rater_size <- max(rater_array, na.rm = TRUE) + 1
  rater_embed_tensor <- layer_flatten(layer_embedding(object = rater_tensor, input_dim = rater_size, output_dim = rater_embedding_size))

  rated_tensor <- layer_input(shape= dim(rated_array)[-1])
  rated_size <- max(rated_array, na.rm = TRUE) + 1
  rated_embed_tensor <- layer_flatten(layer_embedding(object = rated_tensor, input_dim = rated_size, output_dim = rated_embedding_size))

  concat_tensor <- layer_concatenate(list(rater_embed_tensor, rated_embed_tensor))
  interim <- concat_tensor

  for(l in 1:configuration$layers)
  {
    interim <- layer_dense(object = interim, units = unlist(configuration$nodes)[l],
                           kernel_regularizer = regularizer_l1_l2(l1=unlist(configuration$regularization_L1)[l],
                                                                  l2=unlist(configuration$regularization_L2)[l]))

    swish <- function(x, beta = 1){x * k_sigmoid(beta * x)}
    mish <- function(x){x * k_tanh(k_softplus(x))}
    gelu <- function(x){0.5 * x * (1 + k_tanh(sqrt(2/pi) * (x + 0.044715 * x ^ 3)))}
    bent <- function(x){(sqrt(x^2 + 1) - 1) / 2 + x}

    checklist<-c("elu", "relu", "selu", "leaky_relu", "parametric_relu", "thresholded_relu", "softmax", "swish", "mish", "gelu", "bent")###SINC, SINF
    if(unlist(configuration$activations)[l]=="elu") {interim<- layer_activation_elu(object=interim)}
    if(unlist(configuration$activations)[l]=="relu") {interim<- layer_activation_relu(object=interim)}
    if(unlist(configuration$activations)[l]=="leaky_relu") {interim<- layer_activation_leaky_relu(object=interim)}
    if(unlist(configuration$activations)[l]=="parametric_relu") {interim<- layer_activation_parametric_relu(object=interim)}
    if(unlist(configuration$activations)[l]=="thresholded_relu") {interim<- layer_activation_thresholded_relu(object=interim)}
    if(unlist(configuration$activations)[l]=="selu") {interim<- layer_activation_selu(object=interim)}
    if(unlist(configuration$activations)[l]=="softmax") {interim<- layer_activation_softmax(object=interim)}
    if(unlist(configuration$activations)[l]=="swish") {interim<- layer_activation(object=interim, activation = swish)}
    if(unlist(configuration$activations)[l]=="mish") {interim<- layer_activation(object=interim, activation = mish)}
    if(unlist(configuration$activations)[l]=="gelu") {interim<- layer_activation(object=interim, activation = gelu)}
    if(unlist(configuration$activations)[l]=="bent") {interim<- layer_activation(object=interim, activation = bent)}
    if(!(unlist(configuration$activations)[l] %in% checklist)){interim<- layer_activation(object=interim, activation = unlist(configuration$activations)[l])}

    interim<-layer_dropout(object=interim, rate=unlist(configuration$dropout)[l])
    interim<-layer_batch_normalization(object=interim)
  }

  if(skip_shortcut==TRUE)
  {
    reshaped <- layer_dense(object=interim, units = dim(concat_tensor)[-1])
    interim <- layer_add(list(reshaped, concat_tensor))
  }

  if(is.null(output_activation) & task=="regr"){output_activation="linear"} ###DEFAULT VALUE FOR REGR PROBLEMS
  if(is.null(output_activation) & task=="classif"){output_activation="softmax"} ###DEFAULT VALUE FOR CLASSIF PROBLEMS

  if(task=="regr"|task=="classif"){output_tensor <- layer_dense(object = interim, activation = output_activation, units = dim(rating_array)[-1])}

  model <- keras_model(inputs = list(rater_tensor, rated_tensor), outputs = output_tensor)

  ###DEFAULT VALUES FOR MODEL COMPILE
  if(is.null(loss) & task=="regr"){loss="mean_absolute_error"}
  if(is.null(loss) & task=="classif"){loss="categorical_crossentropy"}
  if(is.null(metrics) & task=="regr"){metrics="mean_absolute_error"}
  if(is.null(metrics) & task=="classif"){metrics="categorical_accuracy"}

  compile(object=model, loss = loss, optimizer = optimizer, metrics = metrics)

  repeat_train_loss<-c()
  repeat_train_metric<-c()
  repeat_val_loss<-c()
  repeat_val_metric<-c()

  fold_train_loss<-matrix(nrow=reps, ncol=folds)
  colnames(fold_train_loss) <- paste0("fold_", 1:folds)
  rownames(fold_train_loss) <- paste0("rep_", 1:reps)

  fold_train_metric<-matrix(nrow=reps, ncol=folds)
  colnames(fold_train_metric) <- paste0("fold_", 1:folds)
  rownames(fold_train_metric) <- paste0("rep_", 1:reps)

  fold_val_loss<-matrix(nrow=reps, ncol=folds)
  colnames(fold_val_loss) <- paste0("fold_", 1:folds)
  rownames(fold_val_loss) <- paste0("rep_", 1:reps)

  fold_val_metric<-matrix(nrow=reps, ncol=folds)
  colnames(fold_val_metric) <- paste0("fold_", 1:folds)
  rownames(fold_val_metric) <- paste0("rep_", 1:reps)

  for(r in 1:reps) ### CYCLES FOR CROSSVALIDATION WITH REPETITION
  {
    set.seed(seed+r)
    fold_index <- sample(folds, n_row_train, replace=TRUE)

    for(k in 1:folds)
    {
      rater_fold_k <- subset(rater_train, index=which(fold_index==k), along=1)
      rated_fold_k <- subset(rated_train, index=which(fold_index==k), along=1)
      rating_fold_k <- subset(rating_train, index=which(fold_index==k), along=1)

      rater_fold_non_k <- subset(rater_train, index=which(fold_index!=k), along=1)
      rated_fold_non_k <- subset(rated_train, index=which(fold_index!=k), along=1)
      rating_fold_non_k <- subset(rating_train, index=which(fold_index!=k), along=1)

      ###MODEL FIT
      history <- model %>% fit(list(rater_fold_non_k, rated_fold_non_k), rating_fold_non_k, epochs = epochs, batch_size=batch_size, verbose = verbose,
                               validation_data = list(list(rater_fold_k, rated_fold_k), rating_fold_k), callbacks = list(callback_early_stopping(monitor="val_loss", min_delta=min_delta, patience=floor(epochs*span), restore_best_weights=TRUE)))

      optim_epoch<-which.min(history$metrics[[3]])
      fold_train_loss[r, k] <- history$metrics[[1]][[optim_epoch]]
      fold_train_metric[r, k] <-history$metrics[[2]][[optim_epoch]]
      fold_val_loss[r, k] <- history$metrics[[3]][[optim_epoch]]
      fold_val_metric[r, k] <-history$metrics[[4]][[optim_epoch]]

    }

    repeat_train_loss[r] <- mean(fold_train_loss[r, ], na.rm=TRUE)
    repeat_train_metric[r]<-mean(fold_train_metric[r, ], na.rm=TRUE)
    repeat_val_loss[r] <- mean(fold_val_loss[r, ], na.rm=TRUE)
    repeat_val_metric[r]<-mean(fold_val_metric[r, ], na.rm=TRUE)

  }

  train_loss <- mean(repeat_train_loss, na.rm=TRUE)
  train_metric <- mean(repeat_train_metric, na.rm=TRUE)
  val_loss <- mean(repeat_val_loss, na.rm=TRUE)
  val_metric <- mean(repeat_val_metric, na.rm=TRUE)

  ###TESTING STATS###LIMITED HOLDOUT TO CONTROL OVERFITTING
  history <- model %>% fit(list(rater_train, rated_train), rating_train, epochs = epochs, batch_size=batch_size, verbose = verbose,
                           validation_data = list(list(rater_test, rated_test), rating_test), callbacks = list(callback_early_stopping(monitor="loss",
                                                                                                                                       min_delta=min_delta, patience=floor(epochs*span),restore_best_weights=TRUE))) ###FINAL FIT OVER THE WHOLE DATASET

  results <- model %>% evaluate(list(rater_test, rated_test), rating_test, verbose = 0, batch_size=batch_size)

  test_loss<-results[[1]]
  test_metric<-results[[2]]


  ###PRED FUNCTION
  pred_fun <- function(rater, rated)
  {
    new_rater <- data.matrix(rater)
    colnames(new_rater) <- rater_label

    new_rated <- data.matrix(rated)
    colnames(new_rated) <- rated_label

    new <- list(new_rater, new_rated)

    if(task=="regr")
    {
      prediction <- predict(model, new, batch_size = batch_size)###TOLTO DATA.MATRIX SU DATA
      colnames(prediction) <- "predicted_rating"
    }

    if(task == "classif")
    {
      prediction_prob <- predict(model, new, batch_size = batch_size)
      colnames(prediction_prob) <- paste0("prob_class_", class_names)
      predicted_rating <- factor(class_names[apply(prediction_prob, 1, which.max)], levels = class_names)
      prediction <- data.frame(prediction_prob, predicted_rating)
    }

    return(prediction)
  }


  test_metrics <- NULL

  if(task=="regr")
  {
    test_prediction <- as.data.frame(pred_fun(rater_test, rated_test))
    test_reference <- as.data.frame(rating_test)
    testing_frame <- as.data.frame(map2(test_reference, test_prediction, ~ cbind(reference=.x, predicted=.y)))
    test_metrics <- unlist(map(list(rmse, mae, mdae, mape, smape, rae, rrse), ~ .x(actual = unlist(test_reference), predicted = unlist(test_prediction))))
    names(test_metrics) <- c("rmse", "mae", "mdae", "mape", "smape", "rae", "rrse")
    ###options(scipen=999)###DISABLING SCIENTIFIC NOTATION
    test_metrics <- round(test_metrics, 3)
  }

  if(task=="classif")
  {
    test_prediction <- pred_fun(rater_test, rated_test)
    reference <- rating_factor[test_index]
    predicted <- test_prediction$predicted_rating

    test_metrics <- c(bac = bac(as.numeric(reference), as.numeric(predicted)),
                      avs = avs(as.numeric(reference), as.numeric(predicted)),
                      avp = avp(as.numeric(reference), as.numeric(predicted)),
                      avf = avf(as.numeric(reference), as.numeric(predicted)),
                      kend = ifelse(sd(as.numeric(predicted))>0, cor(as.numeric(reference), as.numeric(predicted), method="kendall", use="pairwise.complete.obs"), 0),
                      ndcg = Evaluation.NDCG(as.numeric(predicted), as.numeric(reference)))

    testing_frame <- data.frame(predicted = predicted, reference = reference)
  }

  ###PREDICTIVE STAT FOR EACH MODEL
  configuration$train_loss<-train_loss
  configuration$val_loss<-val_loss
  configuration$test_loss<-test_loss
  configuration$train_metric<-train_metric
  configuration$val_metric<-val_metric
  configuration$test_metric<-test_metric

  training_metrics <- list(fold_train_metric=fold_train_metric, fold_val_metric=fold_val_metric,
                           repeat_train_metric=repeat_train_metric, repeat_val_metric=repeat_val_metric,
                           train_metric=train_metric, val_metric=val_metric, test_metric=test_metric)

  ###RECOMMENDATION FUNCTION
  recommend <- function(rater, top_n = 10, decreasing = TRUE)
  {
    if(!is.character(rater)){rater <- as.character(rater)}

    not_rated <- unique(data[, rated_label][data[, rater_label] != rater])

    new <-  data.matrix(expand.grid(rater_dict[[rater]], values(rated_dict, not_rated)))
    colnames(new) <- c(rater_label, rated_label)

    new_rater <- new[, rater_label, drop = FALSE]
    new_rated <- new[, rated_label, drop = FALSE]

    prediction <- pred_fun(new_rater, new_rated)

    if(task=="classif")
    {
      predicted <- data.frame(rater = rater, rated = values(inv_rated, new_rated), predicted_rating = prediction$predicted_rating, probability = apply(prediction[, - (class_num + 1)], 1, max))
      predicted <-  predicted[order(predicted$predicted_rating, predicted$probability, decreasing=c(decreasing, TRUE)), , drop = FALSE]
      recommended <- predicted[1:top_n, , drop = FALSE]
    }

    if(task=="regr")
    {
      predicted <- data.frame(rater = rater, rated = values(inv_rated, new_rated), predicted_rating = as.vector(prediction))
      predicted <-  predicted[order(predicted$predicted_rating, decreasing=decreasing), , drop = FALSE]
      recommended <- predicted[1:top_n, , drop = FALSE]
    }

    rownames(recommended) <- NULL
    return(recommended)
  }

  funct_parameters <- match.call()

  toc(log = TRUE)
  time_log<-paste0(round(parse_number(unlist(tic.log()))/60, 0)," minutes")

  ###COLLECTED RESULTS
  outcome<-list(configuration = configuration, model = model, pred_fun = pred_fun, recommend = recommend,
                history = history, plot = plot(history), training_metrics = training_metrics,
                testing_frame = testing_frame, test_metrics = test_metrics,
                funct_parameters = funct_parameters, time_log = time_log)

  tf$compat$v1$Session$close(sess)
  tf$keras$backend$clear_session

  return(outcome)
}
