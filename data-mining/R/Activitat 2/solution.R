# view current path
getwd()

# load library
library(readr)

# function to get the path by filename
getfilepath <- function (fileName) {
  path <- "Activitat 2/data/"
  show(path)
  return (paste(path,fileName,sep = ""))
}

# read csv
dataset <- read_csv(getfilepath("titanic.csv"))
View(dataset)

# read with delim
dataset2 <- read_delim(getfilepath("titanic.tsv"), delim='\t')
View(dataset2)

# save with ';"
write_csv(dataset2, getfilepath("titanic_v2.tsv"))


# excel library
library(readxl)

# view xls file
xls_file <- read_excel(getfilepath("movies.xls"))
View(xls_file)

# view all sheets
xls_sheets <- excel_sheets(getfilepath("movies.xls"))
View(xls_sheets)

# show specific sheet
xls_file <- read_excel(getfilepath("movies.xls"), sheet = "2010s")
View(xls_file)

# save xls
openxlsx::write.xlsx(xls_file, file = getfilepath("movies_2010v2.xlsx"))

