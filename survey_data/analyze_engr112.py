import os
import csv
from ExperimentData import ExperimentData
import numpy as np
from sklearn import tree
from sklearn.feature_extraction import DictVectorizer
import matplotlib.pyplot as plt
from matplotlib import pylab
from matplotlib.colors import ListedColormap

data_dir = './engr112/'

locations = [ 'Anywhere', 'TAMU', 'ENGR', '112' ]
frequencies = [ 'Never', 'Rarely', 'Sometimes', 'Often' ]
frequency_values = dict()
frequency_values['Never'] = 0
frequency_values['Rarely'] = 1
frequency_values['Sometimes'] = 2
frequency_values['Often'] = 3

attributes = [
                'Anywhere', 'TAMU', 'ENGR', '112',
                'Important','How Important', 'Before', 'After',
                'Share','Medium','Gender','Ethnicity','Slogan'
                'Comments'
             ]
'''
def BuildDecisionTree(experiment_data):
    vec = DictVectorizer()
    featuresets = [ (e.GetMultiple(locations), e.Get('How Important'))
                            for e in experiment_data ]
    featuresets = [ pair for pair in featuresets if all(pair) ]
    features_list = [ features for features,value in featuresets ]
    d = dict()
    #X = [ { frequency : 0. for frequency in frequencies } for _ in features_list ]
    for i, features, d in enumerate(zip(features_list)):
        X[i] =
    X = vec.fit_transform([item[0] for item in featuresets]).toarray()
    #X = [ attributes for (attributes,value) in pairs ]
    Y = [ float(value) for (attributes, value) in pairs ]

    print vec.get_feature_names()
'''


'''
  Old way - testing new way
'''

primes = [3,7,13,23]
MaxScore = lambda: 3 * sum(primes)

def Filter(experiments, properties):
    to_keep = []
    for e in experiments:
        keep = True
        for attribute, value in properties:
            if e.Get(attribute) != value:
                keep = False
                break
        if keep:
            to_keep.append(e)

    return to_keep


def SeenScore(features):
    n,r,s,o = features
    #return 1 + primes[1]*n + primes[0]*r + primes[2]*s + primes[3]*o
    return 1 + primes[1]*n + primes[0]*r + primes[2]*s + primes[3]*o


def Plot(X,y):
    seen_scores = []
    color_map = ListedColormap(['b','c','y','r'])
    colors = []
    areas = []
    for x in X:
        n,r,s,o = x
        seen_scores.append(1. * SeenScore(x)/MaxScore())
        colors.append(max(x))

        area = -1

        if max(x) == o:
            area = 650
        elif  max(x) == s:
            area = 500
        elif max(x) == r:
            area = 350
        elif max(x) == n:
            area = 200

        areas.append(area)

    sct = plt.scatter(seen_scores, y,s=areas, c=colors,cmap=color_map)
    cbar = plt.colorbar(sct)
    #plt.sct.set_alpha(0.75)

    cbar.ax.get_yaxis().set_ticks([])
    for j, lab in enumerate(['$Never$','$Rarely$','$Sometimes$','$Often$']):
        cbar.ax.text(1, (2 * j + 1) / 8.0, lab, ha='left', va='center')

    max_score = 3 * sum(primes)
    plt.axis([0,1,0,11])
    plt.xlabel('Experience Score with Bias in Engineering')
    plt.ylabel('Rating of Importance of Bias in Engineering')
    plt.show()


def PlotSurveyData(features, ratings):
    Plot(features, ratings)




def PlotImportancePredictions(predictions):
    features,ratings = zip(*predictions)
    Plot(features, ratings)


def FillMissingData(experiment_data):
    for e in experiment_data:
        if not e.HasValues(locations):
            continue
        where_list = e.GetMultiple(locations)
        where_list = [ frequency_values[where] for where in where_list ]
        if not any(where_list):
            if not e.Get('How Important'):
                e.Set('How Important', 1)

def BuildDecisionTree(experiment_data):
    pairs = [ (e.GetMultiple(locations), e.Get('How Important'))
                            for e in experiment_data ]
    pairs = [ pair for pair in pairs if all(pair) ]
    X = [ attributes for (attributes,value) in pairs ]
    X = [ [ frequency_values[frequency] for frequency in x ] for x in X ]
    Y = [ float(value) for (attributes, value) in pairs ]

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X,Y)
    predictions = []
    for x1 in range(4):
        for x2 in range(4):
            for x3 in range(4):
                for x4 in range(4):
                    x = [ x1, x2, x3, x4 ]
                    prediction = clf.predict([x])
                    predictions.append((x,prediction))

    predictions = sorted(predictions, key=lambda (x,prediction): prediction)

    for (x, prediction) in predictions:
        print x, ' : ', prediction

    #tree.export_graphviz(clf, out_file='tree.dot')
    print X
    #PlotImportancePredictins((X,Y))
    PlotImportancePredictions(predictions)
    PlotSurveyData(X,Y)




def Summary(experiment_data, where):
    experiment_data = sorted(experiment_data, key=lambda e: e.Get(where))
    print where

    for frequency in ['Never', 'Rarely', 'Sometimes', 'Often']:
        separated = [ e for e in experiment_data if e.Get(where) == frequency ]
        importance = [ float(e.Get('How Important')) for e in separated if e.HasValues(['How Important']) ]
        importance_avg = np.mean(importance)
        print '  %s: %.2f' % (frequency, importance_avg)


def SeenSummaries(experiment_data):
    Summary(experiment_data, 'Anywhere')
    Summary(experiment_data, 'TAMU')
    Summary(experiment_data, 'ENGR')
    Summary(experiment_data, '112')


def PlotTrends(experiment_data):
    X = ExperimentData.GetMultipleFrom(experiment_data, locations)
    importance = ExperimentData.GetFrom(experiment_data, 'How Important')
    before = ExperimentData.GetFrom(experiment_data, 'Before')
    after = ExperimentData.GetFrom(experiment_data, 'After')
    zipped = zip(X, importance, before)
    for elem in zipped:
        print elem

    zipped = [ elem for elem in zipped if all(elem) ]

    X,y,areas = zip(*zipped)
    X = [ [ frequency_values[frequency] for frequency in x ] for x in X ]
    y = map(float, y)
    areas = [ frequency_values[a]*150 + 150 for a in areas ]

    seen_scores = []
    color_map = ListedColormap(['b','c','y','r'])
    colors = []
    for x in X:
        n,r,s,o = x
        seen_scores.append(1. * SeenScore(x)/MaxScore())
        colors.append(max(x))

    sct = plt.scatter(seen_scores, y,s=areas, c=colors,cmap=color_map)
    cbar = plt.colorbar(sct)
    #plt.sct.set_alpha(0.75)

    cbar.ax.get_yaxis().set_ticks([])
    for j, lab in enumerate(['$Never$','$Rarely$','$Sometimes$','$Often$']):
        cbar.ax.text(1, (2 * j + 1) / 8.0, lab, ha='left', va='center')

    max_score = 3 * sum(primes)
    plt.axis([0,1,0,11])
    plt.xlabel('Experience Score with Bias in Engineering')
    plt.ylabel('Rating of Importance of Bias in Engineering')
    plt.show()

    #pairs = [ pair for pair in pairs if all(pair) ]
    #X = [ attributes for (attributes,value) in pairs ]
    #X = [ [ frequency_values[frequency] for frequency in x ] for x in X ]
    #y = [ float(value) for (attributes, value) in pairs ]


def Analyze(experiment_data):
    #SeenSummaries(experiment_data)
    #BuildDecisionTree(experiment_data)
    PlotTrends(experiment_data)

def main():
    data_filenames = [ fname for fname in os.listdir(data_dir)
                                    if fname.endswith('.csv') ]

    experiment_data = []
    for filename in reversed(data_filenames):

        print 'File:', filename
        with open(data_dir + filename, 'rU') as csvfile:
            print 'Analyzing',filename
            csvreader = csv.reader(x.replace('\0','') for x in csvfile)
            rows = []
            for row in csvreader:
                rows.append(row)

            for row in rows[2:]:
                experiment_data.append(ExperimentData(attributes, row))

        print 'Experiment Data: ',len(experiment_data)
    #FillMissingData(experiment_data)
    print len(experiment_data)
    filter_properties = [
                            ('112', 'Rarely')
                        ]
    experiment_data = Filter(experiment_data, filter_properties)
    print len(experiment_data)

    Analyze(experiment_data)


main()
