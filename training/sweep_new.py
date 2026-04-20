from clearml import Task
from clearml.automation import HyperParameterOptimizer
from clearml.automation.optuna import OptimizerOptuna
from clearml.automation.parameters import (
    UniformParameterRange,
    LogUniformParameterRange,
    DiscreteParameterRange,
)

# Initialize the Sweeper Task
task = Task.init(
    project_name="ALIKED-Finetuning",
    task_name="punisher-v2-sweeper-large-10epoch-bg-big-data",
    task_type=Task.TaskTypes.optimizer,
)

optimizer = HyperParameterOptimizer(
    base_task_id="63d7d0d90d7b49a7b5fa84488c493172",
    optimizer_class=OptimizerOptuna,
    execution_queue="default",
    hyper_parameters=[
        LogUniformParameterRange(
            name="General/lr",
            min_value=-4.8,
            max_value=-3.6,
        ),
        UniformParameterRange(
            name="General/pos_weight",
            min_value=1.0,
            max_value=5.0,
        ),
        UniformParameterRange(
            name="General/bg_weight",
            min_value=4.0,
            max_value=10.0,
        ),
        UniformParameterRange(
            name="General/lambda_desc",
            min_value=0.5,
            max_value=2.0,
        ),
        DiscreteParameterRange(name="General/epochs", values=[10]),
        DiscreteParameterRange(name="General/batch_size", values=[2]),
        DiscreteParameterRange(name="General/recon_interval", values=[0]),
        DiscreteParameterRange(
            name="General/recon_gpu", values=[None]
        ),  # None = safe_gpu allocates dynamically
        DiscreteParameterRange(name="General/num_gpus", values=[2]),
    ],
    objective_metric_title="reconstruction",
    objective_metric_series="mini_score",
    objective_metric_sign="max_global",
    total_max_jobs=50,
    max_iteration_per_job=10,
    max_number_of_concurrent_tasks=1,
    optimizer_kwargs={
        "sampler": "TPESampler",
        "pruner": "MedianPruner",
        "n_warmup_steps": 4,
        "n_startup_trials": 10,
    },
)

optimizer.start()
optimizer.wait()
optimizer.stop()
