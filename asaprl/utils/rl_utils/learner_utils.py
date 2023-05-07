from ding.worker import SampleSerialCollector, InteractionSerialEvaluator, BaseLearner, NaiveReplayBuffer
from ding.utils import build_logger, EasyTimer, import_module, LEARNER_REGISTRY, get_rank, get_world_size

@LEARNER_REGISTRY.register('SkillLearner')
class SkillLearner(BaseLearner):
    def train(self, data: dict, envstep: int = -1) -> None:
        """
        Overview:
            Given training data, implement network update for one iteration and update related variables.
            Learner's API for serial entry.
            Also called in ``start`` for each iteration's training.
        Arguments:
            - data (:obj:`dict`): Training data which is retrieved from repaly buffer.

        .. note::

            ``_policy`` must be set before calling this method.

            ``_policy.forward`` method contains: forward, backward, grad sync(if in multi-gpu mode) and
            parameter update.

            ``before_iter`` and ``after_iter`` hooks are called at the beginning and ending.
        """
        assert hasattr(self, '_policy'), "please set learner policy"
        self.call_hook('before_iter')

        # Forward
        log_vars = self._policy.forward(data, self.train_iter)

        # Update replay buffer's priority info
        if isinstance(log_vars, dict):
            priority = log_vars.pop('priority', None)
        elif isinstance(log_vars, list):
            priority = log_vars[-1].pop('priority', None)
        else:
            raise TypeError("not support type for log_vars: {}".format(type(log_vars)))
        if priority is not None:
            replay_buffer_idx = [d.get('replay_buffer_idx', None) for d in data]
            replay_unique_id = [d.get('replay_unique_id', None) for d in data]
            self.priority_info = {
                'priority': priority,
                'replay_buffer_idx': replay_buffer_idx,
                'replay_unique_id': replay_unique_id,
            }
        # Discriminate vars in scalar, scalars and histogram type
        # Regard a var as scalar type by default. For scalars and histogram type, must annotate by prefix "[xxx]"
        self._collector_envstep = envstep
        if isinstance(log_vars, dict):
            log_vars = [log_vars]
        for elem in log_vars:
            scalars_vars, histogram_vars = {}, {}
            for k in list(elem.keys()):
                if "[scalars]" in k:
                    new_k = k.split(']')[-1]
                    scalars_vars[new_k] = elem.pop(k)
                elif "[histogram]" in k:
                    new_k = k.split(']')[-1]
                    histogram_vars[new_k] = elem.pop(k)
            # Update log_buffer
            self._log_buffer['scalar'].update(elem)
            self._log_buffer['scalars'].update(scalars_vars)
            self._log_buffer['histogram'].update(histogram_vars)

            self.call_hook('after_iter')
            self._last_iter.add(1)

        return log_vars