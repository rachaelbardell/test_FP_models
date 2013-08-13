library(stringr)
library(RTextTools)
library(caret)

tweets <- read.csv("~/Desktop/test_FP_models/fp_tweets.csv", stringsAsFactors=F, row.names=NULL)

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

confusionMatrix(manual_class$manual_class, results$SLDA_LABEL)

summaryPredictions <- create_scoreSummary(container, results)
# creates a summary with the best label for each document
# determined by highest algorithm certainty and the highest consensus

confusionMatrix(summaryPredictions$BEST_PROB, tweet_text$manual_class, postitive = "1")
# doesn't work... compute statistics manually below

# returns the number of TRUE values
A <- length(which(summaryPredictions$BEST_PROB == "1" & manual_class$manual_class == "1"))
B <- length(which(summaryPredictions$BEST_PROB == "1" & manual_class$manual_class == "0")) # FP
C <- length(which(summaryPredictions$BEST_PROB == "0" & manual_class$manual_class == "1")) # FN
D <- length(which(summaryPredictions$BEST_PROB == "0" & manual_class$manual_class == "0"))

# following the methods of confusionMatrix from the caret package
(A + D) / (A + B + C + D) # Accuracy
A / (A + C) # Sensitivity
D / (D + B) # Specificity
# .6610 Accuracy, 6889 Sensitivity, .6438 Specificity 