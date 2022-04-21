from methods import *
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

# Generate dataset
X, y = dataset()
XX = probers()

# Prepare classifier
clf = GaussianNB(priors=(.5,.5),
                 var_smoothing=1e-9)

# Train
clf.fit(X, y)

# Tell
print('class_count', clf.class_count_)
print('class_prior', clf.class_prior_)
print('classes', clf.classes_)
print('epsilon_', clf.epsilon_)
print('n_features_in_', clf.n_features_in_)
print('var_', clf.var_)
print('theta_', clf.theta_)

print(clf.get_params)

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

ax[1,2].imshow(yy_pred.reshape(q,q), origin='lower', cmap='bwr')
ax[2,2].imshow(yy_pp[:,1].reshape(q,q), origin='lower', cmap='bwr')

# Classifier plot
ax[0,2].scatter(*X.T, color='black', s=.1)
ax[0,2].scatter(*clf.theta_.T, c=clf.classes_, cmap='bwr')
ax[0,2].scatter(*(clf.var_+clf.theta_).T, c=clf.classes_, cmap='bwr', marker='x')
ax[0,2].grid(ls=":")

for a in ax[:,:2].ravel():
    a.grid(ls=":")
    a.set_xlim(-r,r)
    a.set_ylim(-r,r)

plt.tight_layout()
plt.savefig('foo.png')
