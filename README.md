# BroccoliHead3D.ipynb

The notebook archive for destructive broccoli analysis

# 230424

* Remove model noises for `6-3` origial model, but the algorithms does not work well on this special rounded shape
* `8-24` model fail to build and missing

# 230407

* Fix the centroid position error (find by 1-31, the error easydcp ploting of ellipse)
  * cy, cx = regionprops(binary_img).  # flip x and y
  * orientiation -= 90 degrees.   # rotate 90 degrees
* Using min_bounding_box center as mesh Support continue mode, no need to run from the start when get errormove center

# 230406

* Add config for command control
  * By default, `python utils230406.py` only run the first on as test
  * `python utils230406.py --test bro_id` support run specific broccoli using the given id
  * `python utils230406.py --all` run all the broccoli
  * Support continue mode, `python utils230406 --all --continue`, no need to run from the start when get error
