#Logic behind LOTUS https://www.ijcai.org/proceedings/2023/843
anchor_dataset = FastICA().fit_transform(anchor_dataset)
geom_xx = pointcloud.PointCloud(anchor_dataset)
costs = []
for dataset in self.meta_data_obj.datasets:
    dataset = FastICA().fit_transform(dataset['X_train'])
    geom_yy = pointcloud.PointCloud(dataset)
    prob = ott.problems.quadratic.quadratic_problem.QuadraticProblem(geom_xx, geom_yy)
    solver = gromov_wasserstein.GromovWasserstein(rank=6)
    ot_gwlr = solver(prob)
    cost = ot_gwlr.costs[ot_gwlr.costs > 0][-1]
