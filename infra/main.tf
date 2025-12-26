provider "aws" {
  region = "us-east-1"
}

resource "aws_instance" "rag" {
  ami           = "ami-0c02fb55956c7d316" # Amazon Linux 2023 (us-east-1)
  instance_type = "t3.micro"
  key_name      = var.key_name

  user_data = <<-EOF
    #!/bin/bash
    yum update -y
    yum install docker -y
    systemctl start docker
    systemctl enable docker
    usermod -aG docker ec2-user
  EOF

  vpc_security_group_ids = [aws_security_group.rag.id]

  tags = {
    Name = "rag-free-tier"
  }
}

resource "aws_security_group" "rag" {
  name = "rag-sg"

  ingress {
    from_port   = 7860
    to_port     = 7860
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["YOUR_IP/32"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
