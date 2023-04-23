#-------------------------------------------------------------------------
# AUTHOR: Austin Celestino
# FILENAME: bagging_random_forest.py
# SPECIFICATION: Creating a single decision tree, combining multiple decision trees, and creating a Random Forest to recognize digits
# FOR: CS 4210- Assignment #4
# TIME SPENT: 4 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn import tree
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
import csv

dbTraining = []
dbTest = []
X_training = []
y_training = []
classVotes = [] #this array will be used to count the votes of each classifier
sampleNumber = 0

#reading the training data from a csv file and populate dbTraining
#--> add your Python code here
with open("optdigits.tra", "r") as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        sampleNumber += 1
        holder = []
        for j in row:
            holder.append(int(j))
        dbTraining.append(holder)
        '''
        holder = []
        if i > 0:  # skipping the header
            for j in range(len(row)):
                if row - 1 == int(j):
                    y_training.append(j)
                else:
                    holder.append(int(j))
            X_training.append(holder)
        '''

#reading the test data from a csv file and populate dbTest
#--> add your Python code here
with open("optdigits.tes") as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        holder = []
        for j in row:
            holder.append(int(j))
        dbTest.append(holder)

#inititalizing the class votes for each test sample. Example: classVotes.append([0,0,0,0,0,0,0,0,0,0])
#--> add your Python code here
for i in range(len(dbTest)):
    classVotes.append([0,0,0,0,0,0,0,0,0,0])

print("Started my base and ensemble classifier ...")

numberTestSamples = len(dbTest)

for k in range(20): #we will create 20 bootstrap samples here (k = 20). One classifier will be created for each bootstrap sample

  bootstrapSample = resample(dbTraining, n_samples=len(dbTraining), replace=True)
  #print(bootstrapSample)

  #populate the values of X_training and y_training by using the bootstrapSample
  #--> add your Python code here
  # X_training = bootstrapSample[:-2]
  # y_training = bootstrapSample[-1]
  for i in bootstrapSample:
      holder = []
      for j in range(len(i)):
          if j == len(i) - 1:
              y_training.append(i[j])
          else:
              holder.append(i[j])
      X_training.append(holder)
  #print(X_training)
  #fitting the decision tree to the data
  clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=None) #we will use a single decision tree without pruning it
  clf = clf.fit(X_training, y_training)

  numberRight = 0

  for i, testSample in enumerate(dbTest):

      #make the classifier prediction for each test sample and update the corresponding index value in classVotes. For instance,
      # if your first base classifier predicted 2 for the first test sample, then classVotes[0,0,0,0,0,0,0,0,0,0] will change to classVotes[0,0,1,0,0,0,0,0,0,0].
      # Later, if your second base classifier predicted 3 for the first test sample, then classVotes[0,0,1,0,0,0,0,0,0,0] will change to classVotes[0,0,1,1,0,0,0,0,0,0]
      # Later, if your third base classifier predicted 3 for the first test sample, then classVotes[0,0,1,1,0,0,0,0,0,0] will change to classVotes[0,0,1,2,0,0,0,0,0,0]
      # this array will consolidate the votes of all classifier for all test samples
      #--> add your Python code here
      x_test = []
      y_test = []
      for j in range(len(testSample)):
          if j == len(testSample) - 1:
              y_test.append(testSample[j])
          else:
              x_test.append(testSample[j])
      prediction = clf.predict([x_test])
      classVotes[i][prediction[0]] += 1

      if k == 0: #for only the first base classifier, compare the prediction with the true label of the test sample here to start calculating its accuracy
         #--> add your Python code here
        #print("iterated through")
        if prediction == y_test:
            numberRight += 1


  if k == 0: #for only the first base classifier, print its accuracy here
     #--> add your Python code here
     accuracy = numberRight / numberTestSamples
     print("Finished my base classifier (fast but relatively low accuracy) ...")
     print("My base classifier accuracy: " + str(accuracy))
     print("")

  #now, compare the final ensemble prediction (majority vote in classVotes) for each test sample with the ground truth label to calculate the accuracy of the ensemble classifier (all base classifiers together)
  #--> add your Python code here

#print(classVotes)
numberRight = 0
for i, ensemble in enumerate(classVotes):
    highNumber = 0
    selectedIndex = 0
    for j in range(len(ensemble)):
        if ensemble[j] > highNumber:
            highNumber = ensemble[j]
            selectedIndex = j
    if dbTest[i][64] == selectedIndex:
        numberRight += 1
accuracy = numberRight / numberTestSamples

#printing the ensemble accuracy here
print("Finished my ensemble classifier (slow but higher accuracy) ...")
print("My ensemble accuracy: " + str(accuracy))
print("")

print("Started Random Forest algorithm ...")

#Create a Random Forest Classifier
clf=RandomForestClassifier(n_estimators=20) #this is the number of decision trees that will be generated by Random Forest. The sample of the ensemble method used before

#Fit Random Forest to the training data
clf.fit(X_training,y_training)

#make the Random Forest prediction for each test sample. Example: class_predicted_rf = clf.predict([[3, 1, 2, 1, ...]]
#--> add your Python code here
numberRight = 0
for i, testSample in enumerate(dbTest):
    x_test = []
    y_test = []
    for j in range(len(testSample)):
        if j == len(testSample) - 1:
            y_test.append(testSample[j])
        else:
            x_test.append(testSample[j])
    class_predicted_rf = clf.predict([x_test])
    if class_predicted_rf == y_test:
        numberRight += 1

accuracy = numberRight / numberTestSamples

#compare the Random Forest prediction for each test sample with the ground truth label to calculate its accuracy
#--> add your Python code here

#printing Random Forest accuracy here
print("Random Forest accuracy: " + str(accuracy))

print("Finished Random Forest algorithm (much faster and higher accuracy!) ...")
