library(textcat)
library(stringr)

setwd("~/Documents/test_FP_models/Google_API")

dir <- "~/Documents/test_FP_models/"

# Cory's code to clean the text and use textcat to classify new tweets

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

# read the manually classified tweets
df <- read.csv(paste0(dir, "fp_tweets.csv"), stringsAsFactors=F)

# do the preprocessing to the data. this needs to be done before any prediction
# using the model that is created.
df$text.cleansed <- as.character(sapply(df$text, clean.text))
df <- subset(df, !is.na(manual_class) & text.cleansed != "")
df <- subset(df, !duplicated(text.cleansed))

# train the model using the textcat package
fp.model <- textcat_profile_db(df$text.cleansed, df$manual_class)

# test it out...
textcat("i have the food poisoning", fp.model)

#---------------------------------------------------------------------
#---------------------------------------------------------------------

# My code to train/test new models

# train the model with 75% of the manually classified tweets
sample <- sample(1:nrow(df), 0.70*nrow(df))

train <- df[sample,]
pred <- df[-sample,]

fp.model <- textcat_profile_db(train$text.cleansed, train$manual_class)

# use the remaining 25% of the manually classified tweets for prediction
pred <- sapply(pred$text.cleansed, function(x)textcat(x, fp.model))

#-------------------------------------------------------------------------
# Google Prediction API
# data to be exported and used in Google Prediction API
export <- subset(df, select=c("manual_class, text.cleansed"))

# prediction works best if the factors are not integers
export$manual_class[export$manual_class == "1"] <- "T"
export$manual_class[export$manual_class == "0"] <- "F"

# train the Google prediction API with the same 75% as we did with the fp.model
# pred the same remaining 25% with the google prediction API and compare
# make sure there are no column or row names, and that the manual_class column is first
export_train <- export[sample,]
export_pred <- export[-sample,]
write.csv(export_train, file = 'exported_train_FP.csv', row.names = F, col.names=FALSE)
write.csv(export_pred, file = 'exported_pred_FP.csv', row.names = F, col.names=FALSE)

# import results from GoogleAPI
GoogleAPI <- read.csv(paste0(dir, "Google_API/exported_pred_FP.csv"), stringsAsFactors=F, header = F)

# same stats function as the one in create_model.R
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

stats_fp.model <- stats(pred, manual_class)
# .7209 Accuracy, .7778 Sensitivity, .6667 Specificity
stats_Google <- stats(V1, V2)
# 