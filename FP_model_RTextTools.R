library(stringr)
library(RTextTools)
library(caret)

dir <- "~/Documents/test_FP_models/"

tweets <- read.csv(paste0(dir, "fp_tweets.csv"), stringsAsFactors=F, row.names=NULL)

tweet_text <- subset(tweets, select=c("text", "manual_class"), !is.na(manual_class) & text != "" & !duplicated(text), row.names=NULL)

# function clean.text from create_model.R
tweet_text$text <- as.character(sapply(tweet_text$text, clean.text))

text <- create_matrix(tweet_text$text, language="english",
                      removePunctuation=TRUE, 
                      toLower=TRUE, 
                      removeStopwords=TRUE, 
                      removeNumbers=TRUE, 
                      stripWhitespace=TRUE,
                      stemWords=TRUE)

# train model with RTextTool

# train with 75% and predict/test with the remaining 25%
container <- create_container(text, tweet_text$manual_class, trainSize=1:355, testSize = 356:473, virgin = FALSE)

FP_models <- train_models(container, algorithms = c("SLDA", "GLMNET", "MAXENT", "RF"), verbose = TRUE) 
# other algorithms are tree and boosting. did not perform well

results <- classify_models(container, FP_models)
analytics <- create_analytics(container, results)
summary(analytics)
# without removing stop words is about the same 

test_set <- tweet_text[356:473,]

confusionMatrix(test_set$manual_class, results$SLDA_LABEL, positive = "1")

summaryPredictions <- create_scoreSummary(container, results)
# creates a summary with the best label for each document
# determined by highest algorithm certainty and the highest consensus

confusionMatrix(test_set$manual_class, summaryPredictions$BEST_PROB,postitive = "1")
# doesn't work... compute statistics manually below

stats <- function(predictions, manual_class){
  # pred is the column name for the column of predicted classifications
  # man is the column name for the manual classified column
  
  # following the labels of confusionMatrix from the caret package
  # return the number of TRUE values
  A <- length(which(manual_class == "1" & predictions == "1"))
  B <- length(which(manual_class == "0" & predictions == "1"))
  C <- length(which(manual_class == "1" & predictions == "0"))
  D <- length(which(manual_class == "0" & predictions == "0"))
  # Type I error: food poisoning tweet that model prediction classifies as 0
  # Type II error: non-food poisoning tweet that model prediction classifies as 1
  # A and D are correct classifications
  # C is false negatives and B is false postivies
  
  Accuracy <- (A + D) / (A + B + C + D) # Accuracy
  Sensitivity <- A / (A + C) # Sensitivity
  Specificity <- D / (D + B) # Specificity
  stats <- list(Accuracy, Sensitivity, Specificity)
  names(stats) <- c("Accuracy", "Sensitivity", "Specificity")
  
  return(stats)
}

stats_RTextTools <- stats(test_set$manual_class, summaryPredictions$BEST_PROB)
# .6610 Accuracy, .5438 Sensitivity, .7705 Specificity 