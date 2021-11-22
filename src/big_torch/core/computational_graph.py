from ..layers.layer_mixins import ParamManagerMixin
from ..layers.abstract import GradientInputResolver, ParametrizedObject


class ComputationalComponent:
    LAST_ID = 0

    def __init__(self) -> None:
        self.dependencies = []
        self.dependency_handlers = []
        self.slaves = []

        self.id = ComputationalComponent.LAST_ID
        ComputationalComponent.LAST_ID += 1

    def get_layers(self):
        raise NotImplementedError()

    def subscribe(self, dependencies):
        raise NotImplementedError()

    def get_handler(self, idx):
        raise NotImplementedError()


class LayerComponent(ComputationalComponent, ParametrizedObject, GradientInputResolver):
    def __init__(self, layer):
        self.layer = layer

        if isinstance(layer, ParametrizedObject):
            self._parametrized_layers = [layer]

        super().__init__()

        self.output_names = self.layer.output_names
        self.outputs = [
            f"{self.id}_{self.layer.__class__}_{out_name}"
            for out_name in self.output_names
        ]

    def __getitem__(self, key):
        if isinstance(key, int):
            return key, self, self.outputs[key]
        elif isinstance(key, str):
            return self.output_names.index(key), self, self.outputs[self.output_names.index(key)]

    def subscribe(self, dependencies):
        if isinstance(dependencies, ComputationalComponent):
            dependencies = [dependencies[0]]
        for idx, component, dependency_name in dependencies:
            self.dependencies.append(dependency_name)
            self.dependency_handlers.append((component, idx))
            component.slaves.append(self)

    def _fwd_pass(self, X):
        return self.layer._fwd_pass(X)

    def _bwd_pass(self, X, d_out):
        return self.layer._bwd_pass(X, d_out)

    def get_layers(self):
        return [self.layer]

    def get_handler(self, idx):
        return self.layer.get_handler(idx)

    def get_context(self):
        return self.layer.get_context()

    def change(self, step, eta):
        return self.layer.change(step, eta)

    def apply(self, func, context=None):
        return self.layer.apply(func, context)

    def blank(self):
        return self.layer.blank()

    def average(self, gradients_list):
        return self.layer.average(gradients_list)

    def binary_operation(self, lhs, rhs, operation):
        return self.layer.binary_operation(lhs.layer, rhs.layer, operation)


class ComputationalGraph(ComputationalComponent, ParamManagerMixin):
    def __init__(self, input_components, output_components):
        self.input_components = input_components
        self.output_components = output_components

        self.unrolled = False
        self._execution_context = {
            "gradient_context": None,
            "inputs": None,
            "gradients": None,
        }

        super().__init__()

        self.outputs = []
        self.output_names = []
        self.excluded_id = []

        for in_component in self.input_components:
            if isinstance(in_component, ComputationalGraph):
                self.dependencies += in_component.dependencies
            elif isinstance(in_component, LayerComponent):
                self.dependencies += in_component.dependencies
            else:
                raise ValueError(
                    f"ComputationalComponent class was expected, but got {type(in_component)} "
                )

        for out_component in self.output_components:
            if isinstance(out_component, ComputationalGraph):
                self.outputs += out_component.outputs
            elif isinstance(out_component, LayerComponent):
                self.outputs += out_component.outputs
            else:
                raise ValueError(
                    f"ComputationalComponent class was expected, but got {type(out_component)} "
                )

    def clear_execution_context(self):
        self._execution_context = {
            "gradient_context": None,
            "inputs": None,
            "gradients": None,
            "param_grad": None,
        }

    def unroll(self):
        context_set = set()

        def _context_ready(component: ComputationalComponent):
            return set(component.dependencies).issubset(context_set)

        def _checked_dfs(component: ComputationalComponent) -> list:
            _current_elements = []
            if _context_ready(component):
                if isinstance(component, LayerComponent):
                    _current_elements += [component]
                if isinstance(component, ComputationalGraph):
                    _current_elements += component.unroll()
                context_set.update(set(component.outputs))
                for slave in component.slaves:
                    _current_elements += _checked_dfs(slave)
            return _current_elements

        self.ordered_component_list: list = []
        for component in self.input_components:
            context_set.update(component.dependencies)
            self.ordered_component_list += _checked_dfs(component)
        self.ordered_component_list = list(
            filter(
                lambda component: not component.id in self.excluded_id,
                self.ordered_component_list,
            )
        )

        self._ordered_layer_list = []
        self._parametrized_layers = []
        for component in self.ordered_component_list:
            current_layer = component.get_layers()
            self._ordered_layer_list += current_layer
            if isinstance(current_layer[0], ParametrizedObject):
                self._parametrized_layers += current_layer

        self.unrolled = True
        return self.ordered_component_list

    def _store_execution_context(self, category, names, data, handlers=None):
        if category == "gradients":
            storage = self._execution_context[category]
            for handler_src, name, frame in zip(handlers, names, data):
                context = storage.get(name, None)
                if not context is None:
                    handler = handler_src[0].get_handler(handler_src[1])
                    context = handler([context, frame])
                else:
                    context = frame
                storage[name] = context
        else:
            self._execution_context[category].update(
                {name: frame for name, frame in zip(names, data)}
            )

    def _fetch_execution_context(self, category, names):
        return tuple(self._execution_context[category][name] for name in names)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.output_components[key]
        elif isinstance(key, str):
            raise NotImplementedError("Key mapping is not implemented")

    def subscribe(self, dependencies):
        for idx, component, dependency_name in dependencies:
            self.dependencies.append(dependency_name)
            self.dependency_handlers.append((component.get_handlers, idx))
            component.slaves.append(self)

    def _fwd_pass(self, X):

        self._execution_context["inputs"] = {}
        self._execution_context["gradient_context"] = {}

        for idx, input in enumerate(self.input_components):
            if len(input.dependencies) == 1:
                self._execution_context["inputs"][input.dependencies[0]] = X[idx]
            else:
                self._execution_context["inputs"].update(
                    {
                        name: X[idx][subidx]
                        for subidx, name in enumerate(input.dependencies)
                    }
                )

        for component in self.ordered_component_list:
            input = self._fetch_execution_context("inputs", component.dependencies)

            out, component_context = component._fwd_pass(
                input[0] if len(component.dependencies) == 1 else input
            )

            self._store_execution_context(
                "inputs",
                component.outputs,
                out if len(component.outputs) > 1 else tuple([out]),
            )
            self._store_execution_context(
                "gradient_context", [component.id], [component_context]
            )

        return (
            self._fetch_execution_context("inputs", self.outputs),
            self._execution_context["gradient_context"],
        )

    def _bwd_pass(self, X, d_out):

        if self._execution_context["gradient_context"] is None:
            if X is None:
                raise ValueError("_bwd_pass cannot execute without context argument")
            self._execution_context["gradient_context"] = X

        self._execution_context["gradients"] = {}
        self._execution_context["param_grad"] = []

        for idx, output in enumerate(self.output_components):
            _outputs = output.outputs
            if len(_outputs) == 1:
                handler = output.get_handler(0)
                prev_grad = self._execution_context["gradients"].get(_outputs[0], None)
                if not prev_grad is None:
                    prev_grad = handler([prev_grad, d_out[idx]])
                else:
                    prev_grad = d_out[idx]
                self._execution_context["gradients"][_outputs[0]] = prev_grad
            else:
                for subidx, name in enumerate(_outputs):
                    handler = output.get_handler(subidx)
                    prev_grad = self._execution_context["gradients"].get(name, None)
                    if not prev_grad is None:
                        prev_grad = handler([*prev_grad, *d_out[idx][subidx]])
                    else:
                        prev_grad = d_out[idx][subidx]
                    self._execution_context["gradients"][name] = prev_grad

        for component in reversed(self.ordered_component_list):
            d_out = self._fetch_execution_context("gradients", component.outputs)
            # print(f"""Component: {component.outputs}
            #     d_out: {d_out}""")

            grad_context = self._fetch_execution_context(
                "gradient_context", [component.id]
            )

            d_out, param_grads = component._bwd_pass(
                grad_context[0] if len(grad_context) == 1 else grad_context,
                d_out[0] if len(component.outputs) == 1 else d_out,
            )
            self._store_execution_context(
                "gradients",
                component.dependencies,
                [d_out] if len(component.dependencies) == 1 else d_out,
                handlers=component.dependency_handlers,
            )

            if isinstance(component.get_layers()[0], ParametrizedObject):
                self._execution_context["param_grad"].append(param_grads)

        return (
            self._fetch_execution_context("gradients", self.dependencies),
            self._execution_context["param_grad"],
        )

    def get_layers(self):
        try:
            return getattr(self, "_ordered_layer_list", None)
        except:
            raise RuntimeError("Component graph need to be unrolled")
