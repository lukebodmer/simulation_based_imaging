{
  description = "Simulation Based Imaging - numerical solvers, inverse models, and web visualizations";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    custom-nixpkgs.url = "github:lukebodmer/custom_nixpkgs";
  };

  outputs = { self, nixpkgs, custom-nixpkgs, ... }:
    let
      system = "x86_64-linux";

      pkgs = import nixpkgs {
        inherit system;
        overlays = [ custom-nixpkgs.overlays.default ];
        config.allowUnfree = true;
      };

      python = pkgs.python312;
      pythonPackages = python.pkgs;

      # Read pyproject.toml for sbimaging package
      pyproject = builtins.fromTOML (builtins.readFile ./packages/sbimaging/pyproject.toml);
      project = pyproject.project;

      # Build sbimaging as a proper nix package
      sbimagingPackage = pythonPackages.buildPythonPackage {
        pname = project.name;
        inherit (project) version;
        format = "pyproject";
        src = ./packages/sbimaging;

        build-system = with pythonPackages; [
          setuptools
        ];

        propagatedBuildInputs = with pythonPackages; [
          numpy
          scipy
        ];

        doCheck = false;
      };

      # Make sbimaging editable for development
      editableSbimaging = pythonPackages.mkPythonEditablePackage {
        pname = project.name;
        inherit (project) version;
        root = "$PWD/packages/sbimaging/src";
      };

    in
    {
      packages.${system} = {
        sbimaging = sbimagingPackage;
        default = sbimagingPackage;
      };

      devShells.${system}.default = pkgs.mkShell {
        name = "sbi-dev";

        inputsFrom = [
          sbimagingPackage
        ];

        buildInputs = [
          # Editable sbimaging package
          editableSbimaging

          # for pyrobustgasp
          pythonPackages.cppimport # this one is in custom-nixpkgs

          # CUDA and Torch
          pkgs.cudatoolkit
	  pythonPackages.torch-bin

          # GPU support
          pythonPackages.cupy

	  # parsing inputs
          pythonPackages.tomli

          # Visualization
          pythonPackages.pyvista
	  pythonPackages.trame
	  pythonPackages.trame-components
	  pythonPackages.trame-matplotlib
	  pythonPackages.trame-vtk
	  pythonPackages.trame-vuetify

          # Meshing
          pythonPackages.gmsh

          # Django backend
          pythonPackages.django
	  pythonPackages.djangorestframework
	  pythonPackages.django-cors-headers

          # LSP and linting
          pythonPackages.python-lsp-server
          pythonPackages.pyls-flake8
          pythonPackages.flake8
          pythonPackages.ruff

          # Testing
          pythonPackages.pytest

          # Type checking
          pkgs.pyright

          # Node.js for frontend
          pkgs.nodejs_20
          pkgs.nodePackages.npm

          # Additional tools
          pkgs.git
        ];

        shellHook = ''
          export LD_LIBRARY_PATH="${pkgs.cudatoolkit}/lib:$LD_LIBRARY_PATH"
          export PYTHONPATH="$PWD/packages/sbimaging/src:$PWD/control_panel:$PYTHONPATH"
          export VIRTUAL_ENV="SBI Development"

          echo "Simulation Based Imaging development environment"
          echo "Python: $(python --version)"
          echo "Node: $(node --version)"
        '';
      };
    };
}
