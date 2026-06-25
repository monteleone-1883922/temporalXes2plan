(define (domain temporal-test)
  (:requirements :strips :durative-actions)
  (:predicates (started) (done))

  (:durative-action work
    :parameters ()
    :duration (= ?duration 5)
    :condition (at start (not (started)))
    :effect (and (at start (started)) (at end (done))))
)
