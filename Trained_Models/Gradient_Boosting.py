# Add BEFORE training
from genetic_selection import GeneticSelectionCV
from sklearn.ensemble import ExtraTreesClassifier

selector = GeneticSelectionCV(
    estimator=ExtraTreesClassifier(n_estimators=50),
    cv=3,
    verbose=1,
    scoring="f1_macro",
    max_features=1000,
    n_population=50,
    crossover_proba=0.5,
    mutation_proba=0.2,
    n_generations=20,
    crossover_independent_proba=0.5,
    mutation_independent_proba=0.05,
    tournament_size=3,
    n_gen_no_change=5,
    caching=True
)
selector = selector.fit(X_train_res, y_train_res)
X_train_res = selector.transform(X_train_res)
X_val = selector.transform(X_val)