from methods import *
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Generate dataset
X, y = dataset()
XX = probers()

# Prepare classifier
clf = DecisionTreeClassifier(criterion='gini',
                             splitter='random',
                             max_depth=4,
                             random_state=None,
                             class_weight={0:.5,1:.5})

# Train
clf.fit(X, y)

# Prediction
y_pred = clf.predict(X)
y_pp = clf.predict_proba(X)
yy_pred = clf.predict(XX)
yy_pp = clf.predict_proba(XX)

# Presentation
fig, ax = plt.subplots(3,3,figsize=(9,9))

ax[0,0].scatter(*X.T, c=y, cmap='bwr')
ax[1,0].scatter(*X.T, c=y_pred, cmap='bwr')
ax[2,0].scatter(*X.T, c=y_pp[:,1], cmap='bwr')

ax[1,2].imshow(yy_pred.reshape(q,q), origin='lower', cmap='bwr', interpolation='none')
ax[2,2].imshow(yy_pp[:,1].reshape(q,q), origin='lower', cmap='bwr', interpolation='none')

# Classifier plot
app = clf.apply(XX)
ax[0,1].imshow(app.reshape(q,q), origin='lower', cmap='Set1', interpolation='none')

dp = clf.decision_path(X).todense()
ax[1,1].imshow(dp, cmap='binary', interpolation='none')

sdp = np.sort(-dp, axis=1)
ax[0,2].imshow(sdp, cmap='binary', interpolation='none')


for a in ax[:,:1].ravel():
    a.grid(ls=":")
    a.set_xlim(-r,r)
    a.set_ylim(-r,r)

plot_tree(clf, rounded=True, filled=True, ax=ax[2,1])
plt.tight_layout()
plt.savefig('foo.png')
