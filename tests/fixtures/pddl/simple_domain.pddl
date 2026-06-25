(define (domain simple-test)
  (:requirements :strips)
  (:predicates (at-a) (at-b) (at-c))

  (:action go-b
    :parameters ()
    :precondition (at-a)
    :effect (and (not (at-a)) (at-b)))

  (:action go-c
    :parameters ()
    :precondition (at-b)
    :effect (and (not (at-b)) (at-c)))
)
