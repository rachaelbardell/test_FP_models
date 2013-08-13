library(tm)
library(rpart)
library(C50)
library(caret)
library(stringr)

clean.text <- function(text){
  # INPUT: Text to be "cleansed"
  # OUTPUT: Cleansed text
  # USAGE: clean.text(text) will return a string that has the punctuation removed
  #        lower case, and all other text cleaning operations done
  replace.links <- function(text){
    # extract urls from string, only works with t.co links, which all links in
    # twitter are nowadays
    return(str_replace_all(text,
                           ignore.case("http://[a-z0-9].[a-z]{2,3}/[a-z0-9]+"),
                           "urlextracted"))
  }
  remove.word <- function(string, starts.with.char){
    # INPUT:  string is a string to be edited,
    #         starts.with.char is a string or partial string to search and remove
    # OUTPUT: string with words removed
    # USAGE:  remove.word(string, "@") removes words starting with "@"
    #         remove.word(string, "RT") removes RT from string
    word.len <- nchar(starts.with.char)
    list.of.words <- strsplit(string, " ")[[1]]
    # remove ones that start with "starts.with.char"
    list.of.words <- list.of.words[!substring(list.of.words, 1,
                                              word.len)==starts.with.char]
    ret.string <- paste(list.of.words, collapse=" ")
    return(ret.string)
  }
  
  text.cleansed <- tolower(text)
  # remove the string "food poisoning" because every tweet has this in it...
  text.cleansed <- gsub("food poisoning", "", text.cleansed)
  text.cleansed <- replace.links(text.cleansed)
  text.cleansed <- remove.word(text.cleansed, "@")
  text.cleansed <- remove.word(text.cleansed, "rt")
  # replace non-letters with spaces
  text.cleansed <- gsub("[^[:alnum:]]", " ", text.cleansed)
  # remove leading and trailing spaces
  text.cleansed <- gsub("^\\s+|\\s+$", "", text.cleansed)
  # replace multiple spaces next to each other with single space
  text.cleansed <- gsub("\\s{2,}", " ", text.cleansed)
  return(text.cleansed)
}  


# manual_class==1 is good fp tweet, 0 is no good
df <- read.csv("~/Desktop/test_FP_models/fp_tweets.csv", stringsAsFactors=F)
df$text.cleansed <- sapply(df$text, clean.text)
df$manual_class <- factor(df$manual_class)
df <- subset(df, !is.na(manual_class), select=c("text.cleansed", "manual_class"))

corp <- Corpus(DataframeSource(df, encoding="UTF-8"))
corp <- tm_map(corp, FUN=function(x)removeWords(x,stopwords("english")))
# all of this done above via clean.text function

dtm <- DocumentTermMatrix(corp)
dtm.df <- as.data.frame(as.matrix(dtm))
dtm.df <- cbind(manual_class=df$manual_class, dtm.df)
# remove sparse cols
non.sparse.terms <- findFreqTerms(dtm, 5)
dtm.df <- subset(dtm.df, select=c("manual_class", non.sparse.terms))

samp <- sample(nrow(dtm.df), .5*nrow(dtm.df))
dtm.df.train <- dtm.df[samp,]
dtm.df.test <- dtm.df[-samp,]

# Sensitivity - the rate that a good FP tweet is predicted correctly
# Specifity - the rate a bad FP tweet is predicted correctly
# No Information Rate = .6236

ctrl <- trainControl(method="cv", classProbs=FALSE, verboseIter=TRUE)
# classProbs cannot be used with costs

c5Matrix <- matrix(c(0, 1.05, 1, 0), ncol=2)
rownames(c5Matrix) <- levels(dtm.df$manual_class)
colnames(c5Matrix) <- levels(dtm.df$manual_class)
C50 <- train(manual_class ~ ., data=dtm.df.train, method="C5.0", metric="Kappa",
           trControl = ctrl, tuneLength=10, cost = c5Matrix)
pred <- as.data.frame(matrix(nrow = nrow(dtm.df.test)))
pred$C50 <- predict(C50, dtm.df.test)
confusionMatrix(pred$C50, dtm.df.test$manual_class, positive = "1")
# .6578 Accuracy, .1195 Kappa, .1111 Sensitivity, .9878 Specificity

rpartMatrix <- matrix(c(0, 1.05, 1, 0), ncol=2)
rownames(rpartMatrix) <- levels(dtm.df$manual_class)
colnames(rpartMatrix) <- levels(dtm.df$manual_class)
rpart <- train(manual_class ~ ., data=dtm.df.train, method="rpart", metric="Kappa",
           trControl = ctrl, tuneLength=10, parms=list(loss=rpartMatrix))
pred$rpart <- predict(rpart, dtm.df.test)
confusionMatrix(pred$rpart, dtm.df.test$manual_class, positive="1")
# .6502 Accuracy, .1042 Kappa, .1111 Sensitivity, .9756 Specificity

glmnet <- train(manual_class ~ ., data=dtm.df.train, method="glmnet", metric="Kappa",
                trControl = ctrl, tuneLength=10) 
pred$glmnet <- predict(glmnet, dtm.df.test)
confusionMatrix(pred$glmnet, dtm.df.test$manual_class, positive="1")
# .7376 Accuracy, .4657 Kappa, .7677 Sensitivity, .7195 Specificity

# -----

svm <- train(manual_class ~ ., data=dtm.df.train, method="svmRadial", metric="Kappa",
             trControl = ctrl, class.weights = c("1"=1, "0"=2))
pred$svm <- predict(svm, dtm.df.test)
confusionMatrix(pred$svm, dtm.df.test$manual_class, positive = "1")
# .6768 Accuracy, .2049 Kappa, .2323 Sensitivity, .9451 Specificity

knn <- train(manual_class ~ ., data=dtm.df.train, method="knn", metric="Kappa",
            trControl = ctrl, tuneLength=10) # k defualts to 5
pred$knn <- predict(knn, dtm.df.test)
confusionMatrix(pred$knn, dtm.df.test$manual_class, positive="1")
# .652 Accuracy, .368 Kappa, .9596 Sensitivity, .4695 Specificity

tree <- train(manual_class ~ ., data=dtm.df.train, method="bstTree", metric="Kappa",
             trControl = ctrl, tuneLength=10)
pred$tree <- predict(tree, dtm.df.test)
confusionMatrix(pred$tree, dtm.df.test$manual_class, positive="1")
# .7376 Accuracy, .4816 Kappa, .8485 Sensitivity, .6707 Specificity
# takes 30 min to run. using cost = .9 didn't increase specificity

rf <- train(manual_class ~ ., data=dtm.df.train, method="rf", metric="Kappa",
              trControl = ctrl, tuneLength=10)
pred$rf <- predict(rf, dtm.df.test)
confusionMatrix(pred$rf, dtm.df.test$manual_class, positive="1")
# .749 Accuracy, .4957 Kappa, .8182 Sensitivity, .7073 Specificity
# 5 minutes...

# ----

# collecting the results of the models 
resamp <- resamples(list(C50 = C50, rpart = rpart, svm = svm , knn = knn, boostedTree = tree, glmnet= glmnet, rf = rf))
summary(resamp)
diff <- diff(resamp, list("C50", "rpart", "svm", "knn", "boostedTree", "glmnet", "rf"))
diff$statistics # t test comparing pairs of models based on Kappa and Accuracy

# based on Accuracy & Kappa, some models are better than others 
bwplot(diff, what = "differences") 
dotplot(diff, what = "differences")#, metric = "Kappa")

# function to create a column (best) with the best label for each tweet
# determined by highest algorithm consensus based on three models
best_class <- function(predictions) {
              predictions[, 1] <- as.numeric(predictions[, 1])
              predictions[, 2] <- as.numeric(predictions[, 2])
              predictions[, 3] <- as.numeric(predictions[, 3])
        
              predictions$best <- sapply(1:nrow(predictions), function(x) sum(predictions[x,]))
        
              # as.numeric converts factor to numbers 1 (for factor 0) and 2 (for factor 1)
              predictions$best[predictions$best <= 4] <- "0" 
              predictions$best[predictions$best >= 5] <- "1"
        
              predictions$best <- as.factor(predictions$best)
        
              return(predictions)
              }

pred <- pred[, -1] # omit the column of NA's
pred <- best_class(pred)

# results of "best" prediction using the three different models
confusionMatrix(pred$best, dtm.df.test$manual_class, positive = "1")
# .6692 Accuracy, .163 Kappa, .1616 Sensitivity, .9756 Specificity C50, rpart, svm/knn
# .673 Accuracy, .1707 Kappa, .1616 Sensitivity, .9817 Specificity C50, rpart, rf/glmnet
# .6882 Accuracy, .2383 Kappa, .2623 Sensitivity, .9451 Specificity C50/rpart, glmnet, svm
# .6882 Accuracy, .2481 Kappa, .2727 Sensitivity, .9390 Specificity C50, rpart, glmnet, rf, svm

# want more false negatives than false positives (i.e. high specificity)
# prefer more 1's classified as 0's than 0's (junk) getting through as 1's