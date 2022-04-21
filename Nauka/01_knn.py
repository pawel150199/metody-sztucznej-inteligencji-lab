from methods import *
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Generate dataset
X, y = dataset()
XX = probers()

# Prepare classifier
clf = KNeighborsClassifier(n_neighbors=5,
                           weights="uniform",
                           leaf_size=30,
                           p=2,
                           metric="minkowski")

# Train
clf.fit(X, y)

# Prediction
y_pred = clf.predict(X)
y_pp = clf.predict_proba(X)

# Space probing
yy_pred = clf.predict(XX)
yy_pp = clf.predict_proba(XX)

# Presentation
fig, ax = plt.subplots(3,3,figsize=(9,9))

ax[0,0].scatter(*X.T, c=y, cmap='bwr')
ax[1,0].scatter(*X.T, c=y_pred, cmap='bwr')
ax[2,0].scatter(*X.T, c=y_pp[:,1], cmap='bwr')

ax[1,2].imshow(yy_pred.reshape(q,q), origin='lower', cmap='bwr', interpolation='none')
ax[2,2].imshow(yy_pp[:,1].reshape(q,q), origin='lower', cmap='bwr', interpolation='none')


for a in ax[:,:2].ravel():
    a.grid(ls=":")
    a.set_xlim(-r,r)
    a.set_ylim(-r,r)

plt.tight_layout()
plt.savefig('foo.png')
