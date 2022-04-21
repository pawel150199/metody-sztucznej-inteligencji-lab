from methods import *
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Generate dataset
X, y = dataset()
XX = probers()

# Prepare classifier
clf = SVC(kernel="rbf", gamma='auto', shrinking=True, probability=True,
          tol=1e-3, verbose=False, max_iter=100)

# Train
clf.fit(X, y)

# Tell
print(clf.n_support_)
print(clf.intercept_)
print(clf.dual_coef_, clf.dual_coef_.shape)
print(clf.support_, np.unique(y[clf.support_], return_counts=True))
print(clf.support_vectors_)

# Prediction
y_pred = clf.predict(X)
y_dec = clf.decision_function(X)
y_pp = clf.predict_proba(X)
yy_pred = clf.predict(XX)
yy_dec = clf.decision_function(XX)
yy_pp = clf.predict_proba(XX)

# Presentation
fig, ax = plt.subplots(3,3,figsize=(9,9))

ax[0,0].scatter(*X.T, c=y, cmap='bwr')
ax[1,0].scatter(*X.T, c=y_pred, cmap='bwr')
ax[2,0].scatter(*X.T, c=y_dec, cmap='bwr')
ax[2,1].scatter(*X.T, c=y_pp[:,1], cmap='bwr')

ax[1,2].imshow(yy_pred.reshape(q,q), origin='lower', cmap='bwr', interpolation='none')
ax[2,2].imshow(yy_dec.reshape(q,q), origin='lower', cmap='bwr', interpolation='none')
ax[0,2].imshow(yy_pp[:,1].reshape(q,q), origin='lower', cmap='bwr', interpolation='none')


# Classifier plot
ax[0,1].scatter(*X.T, c=y, cmap='bwr', s=1)
ax[0,1].scatter(*clf.support_vectors_.T, c=y[clf.support_], cmap='bwr')

for a in ax[:,:2].ravel():
    a.grid(ls=":")
    a.set_xlim(-r,r)
    a.set_ylim(-r,r)

plt.tight_layout()
plt.savefig('foo.png')
