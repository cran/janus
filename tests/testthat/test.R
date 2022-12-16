skip_if_no_tensorflow <- function() {
  have_tensorflow <- reticulate::py_module_available("tensorflow")
  if (!have_tensorflow)
    skip("tensorflow not available for testing")
}

dummy <- data.frame(rater=sample(30,100,T), rated=sample(100,100,T), rating=sample(5,100,T))

test_that("Correct outcome format and size for base outcome1",
          {
            skip_if_no_tensorflow()
            outcome1 <- janus(dummy, "rating", "rater", "rated", task = "classif", n_samp = 3, n_steps = 1)
            expect_equal(class(outcome1), "list")
            expect_equal(length(outcome1), 3)
            expect_equal(names(outcome1), c("pipeline", "best_model", "time_log"))
            expect_equal(names(outcome1$best_model), c("configuration", "model", "recommend", "plot", "training_metrics", "testing_frame", "test_metrics"))
            expect_equal(class(outcome1$pipeline), "data.frame")
            expect_equal(dim(outcome1$pipeline), c(3, 11))
            expect_equal(class(outcome1$best_model$recommend), "function")
            expect_equal(dim(outcome1$best_model$testing_frame), c(10, 2))
            expect_equal(length(outcome1$best_model$test_metrics),6)
            expect_equal(dim(outcome1$best_model$recommend(as.character(sample(dummy$rater, 1)))), c(10, 4))
          })


test_that("Correct outcome format and size for base outcome2",
          {
            skip_if_no_tensorflow()
            outcome2 <- janus(dummy, "rating", "rater", "rated", task = "regr", n_samp = 1, n_steps = 3, holdout = 0.3)
            expect_equal(class(outcome2), "list")
            expect_equal(length(outcome2), 3)
            expect_equal(names(outcome2), c("pipeline", "best_model", "time_log"))
            expect_equal(names(outcome2$best_model), c("configuration", "model", "recommend", "plot", "training_metrics", "testing_frame", "test_metrics"))
            expect_equal(class(outcome2$pipeline), "data.frame")
            expect_equal(dim(outcome2$pipeline), c(3, 11))
            expect_equal(class(outcome2$best_model$recommend), "function")
            expect_equal(dim(outcome2$best_model$testing_frame), c(30, 2))
            expect_equal(length(outcome2$best_model$test_metrics), 7)
            expect_equal(dim(outcome2$best_model$recommend(as.character(sample(dummy$rater, 1)))), c(10, 3))
          })
