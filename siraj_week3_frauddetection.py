{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "#there were several concepts that I was new to, including the Panda's operation to replace NaN's as well as memory reduction.\n",
    "#Kartik Athale shared some code, so I am employing some of his code here.\n",
    "#as the main goal is to best fit the data, I will mainly be targeting on using grid search to attempt to \n",
    "#optimize LogisticRegression hyperparameters.\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.model_selection import StratifiedShuffleSplit\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import classification_report\n",
    "from sklearn.decomposition import PCA\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.externals import joblib\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from sklearn.metrics import mean_squared_error, r2_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# Read the CSV file\n",
    "train_xact = pd.read_csv('data/train_transaction.csv')\n",
    "test_xact = pd.read_csv('data/test_transaction.csv')\n",
    "train_ident = pd.read_csv('data/train_identity.csv')\n",
    "test_ident = pd.read_csv('data/test_identity.csv')\n",
    "\n",
    "train= pd.merge(train_xact,train_ident,how=\"left\",on=\"TransactionID\")\n",
    "test = pd.merge(test_xact,test_ident,how=\"left\",on=\"TransactionID\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "data loaded...\n"
     ]
    }
   ],
   "source": [
    "print('data loaded...')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mem. usage decreased to 650.48 Mb (66.8% reduction)\n",
      "Mem. usage decreased to 565.37 Mb (66.3% reduction)\n"
     ]
    }
   ],
   "source": [
    "#all credit to Kartik Athale (https://www.kaggle.com/kartikathale/fraud-detection-eda-basic-logistic-regression)\n",
    "\n",
    "\n",
    "def reduce_mem_usage(df, verbose=True):\n",
    "    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']\n",
    "    start_mem = df.memory_usage().sum() / 1024**2    \n",
    "    for col in df.columns:\n",
    "        col_type = df[col].dtypes\n",
    "        if col_type in numerics:\n",
    "            c_min = df[col].min()\n",
    "            c_max = df[col].max()\n",
    "            if str(col_type)[:3] == 'int':\n",
    "                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:\n",
    "                    df[col] = df[col].astype(np.int8)\n",
    "                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:\n",
    "                    df[col] = df[col].astype(np.int16)\n",
    "                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:\n",
    "                    df[col] = df[col].astype(np.int32)\n",
    "                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:\n",
    "                    df[col] = df[col].astype(np.int64)  \n",
    "            else:\n",
    "                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:\n",
    "                    df[col] = df[col].astype(np.float16)\n",
    "                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:\n",
    "                    df[col] = df[col].astype(np.float32)\n",
    "                else:\n",
    "                    df[col] = df[col].astype(np.float64)    \n",
    "    end_mem = df.memory_usage().sum() / 1024**2\n",
    "    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))\n",
    "    return df\n",
    "\n",
    "train=reduce_mem_usage(train)\n",
    "test=reduce_mem_usage(test)\n",
    "\n",
    "del train_xact\n",
    "del test_xact\n",
    "del train_ident\n",
    "del test_ident\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "memory reduction complete\n"
     ]
    }
   ],
   "source": [
    "print('memory reduction complete')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [],
   "source": [
    "#more credit to Kartik\n",
    "\n",
    "#remove columns with > 50% Nulls\n",
    "\n",
    "null_percent = train.isnull().sum()/train.shape[0]*100\n",
    "\n",
    "cols_to_drop = np.array(null_percent[null_percent > 50].index)\n",
    "\n",
    "train = train.drop(cols_to_drop, axis=1)\n",
    "test = test.drop(cols_to_drop,axis=1)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [],
   "source": [
    "### Fill NaNs.   \n",
    "\n",
    "train = train.fillna(-999)\n",
    "test = test.fillna(-999)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [],
   "source": [
    "#split training targets and data.\n",
    "\n",
    "train_y = train['isFraud']\n",
    "train_X = train.drop('isFraud', axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "#preprocessing (docs appear to describe one-hot encoding: taking training and test data field which are non-numeric and encoding a number to show if being used).\n",
    "\n",
    "for f in train_X.columns:\n",
    "    if train_X[f].dtype=='object' or test[f].dtype=='object': \n",
    "        lbl = LabelEncoder()\n",
    "        lbl.fit(list(train_X[f].values) + list(test[f].values))\n",
    "        train_X[f] = lbl.transform(list(train_X[f].values))\n",
    "        test[f] = lbl.transform(list(test[f].values))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [],
   "source": [
    "#unsure the benefit of scaling int values.  it may improve things, but I'm omitting them from this method\n",
    "\n",
    "#https://www.data-blogger.com/2017/06/15/fraud-detection-a-simple-machine-learning-approach/\n",
    "\n",
    "def normalize(df):\n",
    "    \n",
    "    \n",
    "    floats = ['float16', 'float32', 'float64']\n",
    "    for col in df.columns:\n",
    "        col_type = df[col].dtypes\n",
    "        if col_type in floats:\n",
    "            df[col] -= df[col].mean()\n",
    "            df[col] /= df[col].std()\n",
    "    \n",
    "    return df\n",
    "\n",
    "train_X = normalize(train_X)\n",
    "test = normalize(test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "scaling complete\n"
     ]
    }
   ],
   "source": [
    "print('scaling complete')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [],
   "source": [
    "test_X = test\n",
    "test_y = np.ones(test_X.shape[0])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1 1 starting\n",
      "1 1 instantiated\n"
     ]
    },
    {
     "ename": "ValueError",
     "evalue": "Input contains NaN, infinity or a value too large for dtype('float64').",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-73-9bdba426e0ca>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m()\u001b[0m\n\u001b[0;32m      8\u001b[0m         \u001b[0mmodel\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mLogisticRegression\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mpenalty\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;34m'l2'\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mdual\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mFalse\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mtol\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mtol\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mC\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;36m1.0\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mfit_intercept\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mTrue\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mintercept_scaling\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mclass_weight\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mNone\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mrandom_state\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mNone\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0msolver\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;34m'liblinear'\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mmax_iter\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mmax_iter\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mmulti_class\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;34m'ovr'\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mverbose\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mwarm_start\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mFalse\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mn_jobs\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mNone\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      9\u001b[0m         \u001b[0mprint\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtolmag\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mmaxitermag\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m'instantiated'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 10\u001b[1;33m         \u001b[0mmodel\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfit\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtrain_X\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0mtrain_y\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     11\u001b[0m         \u001b[0mprint\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtolmag\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mmaxitermag\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m'fit complete'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     12\u001b[0m         \u001b[0mpred\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mmodel\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mpredict\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtest_X\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mc:\\users\\u773253\\appdata\\local\\programs\\python\\python37\\lib\\site-packages\\sklearn\\linear_model\\logistic.py\u001b[0m in \u001b[0;36mfit\u001b[1;34m(self, X, y, sample_weight)\u001b[0m\n\u001b[0;32m   1282\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1283\u001b[0m         X, y = check_X_y(X, y, accept_sparse='csr', dtype=_dtype, order=\"C\",\n\u001b[1;32m-> 1284\u001b[1;33m                          accept_large_sparse=solver != 'liblinear')\n\u001b[0m\u001b[0;32m   1285\u001b[0m         \u001b[0mcheck_classification_targets\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0my\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1286\u001b[0m         \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mclasses_\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0munique\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0my\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mc:\\users\\u773253\\appdata\\local\\programs\\python\\python37\\lib\\site-packages\\sklearn\\utils\\validation.py\u001b[0m in \u001b[0;36mcheck_X_y\u001b[1;34m(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, warn_on_dtype, estimator)\u001b[0m\n\u001b[0;32m    745\u001b[0m                     \u001b[0mensure_min_features\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mensure_min_features\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    746\u001b[0m                     \u001b[0mwarn_on_dtype\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mwarn_on_dtype\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 747\u001b[1;33m                     estimator=estimator)\n\u001b[0m\u001b[0;32m    748\u001b[0m     \u001b[1;32mif\u001b[0m \u001b[0mmulti_output\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    749\u001b[0m         y = check_array(y, 'csr', force_all_finite=True, ensure_2d=False,\n",
      "\u001b[1;32mc:\\users\\u773253\\appdata\\local\\programs\\python\\python37\\lib\\site-packages\\sklearn\\utils\\validation.py\u001b[0m in \u001b[0;36mcheck_array\u001b[1;34m(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, warn_on_dtype, estimator)\u001b[0m\n\u001b[0;32m    566\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0mforce_all_finite\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    567\u001b[0m             _assert_all_finite(array,\n\u001b[1;32m--> 568\u001b[1;33m                                allow_nan=force_all_finite == 'allow-nan')\n\u001b[0m\u001b[0;32m    569\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    570\u001b[0m     \u001b[0mshape_repr\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0m_shape_repr\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0marray\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mshape\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mc:\\users\\u773253\\appdata\\local\\programs\\python\\python37\\lib\\site-packages\\sklearn\\utils\\validation.py\u001b[0m in \u001b[0;36m_assert_all_finite\u001b[1;34m(X, allow_nan)\u001b[0m\n\u001b[0;32m     54\u001b[0m                 not allow_nan and not np.isfinite(X).all()):\n\u001b[0;32m     55\u001b[0m             \u001b[0mtype_err\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;34m'infinity'\u001b[0m \u001b[1;32mif\u001b[0m \u001b[0mallow_nan\u001b[0m \u001b[1;32melse\u001b[0m \u001b[1;34m'NaN, infinity'\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 56\u001b[1;33m             \u001b[1;32mraise\u001b[0m \u001b[0mValueError\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mmsg_err\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mformat\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtype_err\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mX\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mdtype\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     57\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     58\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mValueError\u001b[0m: Input contains NaN, infinity or a value too large for dtype('float64')."
     ]
    }
   ],
   "source": [
    "mses = []\n",
    "\n",
    "for tolmag in range(1,11):\n",
    "    for maxitermag in range(1,6):\n",
    "        tol = 10**(-tolmag)\n",
    "        max_iter = 10**maxitermag\n",
    "        print(tolmag, maxitermag, 'starting')\n",
    "        model = LogisticRegression(penalty='l2', dual=False, tol=tol, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=max_iter, multi_class='ovr', verbose=0, warm_start=False, n_jobs=None)\n",
    "        print(tolmag, maxitermag, 'instantiated')\n",
    "        model.fit(train_X,train_y)\n",
    "        print(tolmag, maxitermag, 'fit complete')\n",
    "        pred = model.predict(test_X)\n",
    "        print(tolmag, maxitermag, 'prediction complete')\n",
    "        currdata = [tolmag, maxitermag, mean_squared_error(test_y, pred)]\n",
    "        print(currdata)\n",
    "        mses.append(currdata)\n",
    "        \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "finished testing\n"
     ]
    }
   ],
   "source": [
    "print('finished testing')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
